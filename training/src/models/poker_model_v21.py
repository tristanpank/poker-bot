import copy
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from poker_state_v21 import ACTION_COUNT_V21, STATE_DIM_V21


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual
        return self.relu(x)


class PokerDeepCFRNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM_V21, hidden_dim: int = 256, action_dim: int = ACTION_COUNT_V21):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.input_relu = nn.ReLU()
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.block3 = ResidualBlock(hidden_dim)

        self.regret_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_relu(self.input_layer(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        trunk = self._forward_trunk(x)
        return {
            "regret": self.regret_head(trunk),
            "strategy": self.strategy_head(trunk),
        }

    def forward_regret(self, x: torch.Tensor) -> torch.Tensor:
        return self.regret_head(self._forward_trunk(x))

    def forward_strategy(self, x: torch.Tensor) -> torch.Tensor:
        return self.strategy_head(self._forward_trunk(x))

    def clone_cpu(self) -> "PokerDeepCFRNet":
        snapshot = PokerDeepCFRNet(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
        )
        snapshot.load_state_dict(copy.deepcopy(self.state_dict()))
        snapshot.eval()
        snapshot.to("cpu")
        for param in snapshot.parameters():
            param.requires_grad = False
        return snapshot


def _normalize_legal_mask(legal_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if mask.shape[0] != ACTION_COUNT_V21:
        raise ValueError(f"Expected legal mask of length {ACTION_COUNT_V21}, got {mask.shape}")
    mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
    if mask.sum() <= 0.0:
        mask[:] = 1.0
    return mask


def regret_matching(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.shape[0] != ACTION_COUNT_V21:
        raise ValueError(f"Expected logits of length {ACTION_COUNT_V21}, got {logits.shape}")
    mask = _normalize_legal_mask(legal_mask)
    regrets = np.maximum(logits, 0.0) * mask
    total = float(regrets.sum())
    if total <= 1e-8:
        probs = mask / float(mask.sum())
        return probs.astype(np.float32)
    return (regrets / total).astype(np.float32)


def masked_policy(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.shape[0] != ACTION_COUNT_V21:
        raise ValueError(f"Expected logits of length {ACTION_COUNT_V21}, got {logits.shape}")
    mask = _normalize_legal_mask(legal_mask)
    masked = np.full_like(logits, -1e9, dtype=np.float32)
    masked[mask > 0.5] = logits[mask > 0.5]
    max_val = float(masked.max())
    exp_vals = np.zeros_like(masked, dtype=np.float32)
    legal = mask > 0.5
    exp_vals[legal] = np.exp(masked[legal] - max_val)
    denom = float(exp_vals.sum())
    if denom <= 1e-8:
        probs = mask / float(mask.sum())
        return probs.astype(np.float32)
    return (exp_vals / denom).astype(np.float32)


class AdvantageBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = STATE_DIM_V21,
        action_dim: int = ACTION_COUNT_V21,
    ):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.legal_masks = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.targets = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.weights = np.ones(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, legal_mask: np.ndarray, target: np.ndarray, weight: float = 1.0) -> None:
        idx = self.position
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.legal_masks[idx] = np.asarray(legal_mask, dtype=np.float32)
        self.targets[idx] = np.asarray(target, dtype=np.float32)
        self.weights[idx] = float(weight)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, samples) -> int:
        count = 0
        for state, legal_mask, target, weight in samples:
            self.add(state, legal_mask, target, weight)
            count += 1
        return count

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty advantage buffer")
        count = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=count)
        return {
            "states": self.states[indices],
            "legal_masks": self.legal_masks[indices],
            "targets": self.targets[indices],
            "weights": self.weights[indices],
        }

    def __len__(self) -> int:
        return self.size


class StrategyBuffer:
    def __init__(
        self,
        capacity: int = 2_000_000,
        state_dim: int = STATE_DIM_V21,
        action_dim: int = ACTION_COUNT_V21,
    ):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.legal_masks = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.targets = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.weights = np.ones(self.capacity, dtype=np.float32)
        self.size = 0
        self.inserts = 0

    def add(self, state: np.ndarray, legal_mask: np.ndarray, target: np.ndarray, weight: float = 1.0) -> None:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = random.randint(0, self.inserts)
            if idx >= self.capacity:
                self.inserts += 1
                return
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.legal_masks[idx] = np.asarray(legal_mask, dtype=np.float32)
        self.targets[idx] = np.asarray(target, dtype=np.float32)
        self.weights[idx] = float(weight)
        self.inserts += 1

    def extend(self, samples) -> int:
        count = 0
        for state, legal_mask, target, weight in samples:
            self.add(state, legal_mask, target, weight)
            count += 1
        return count

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty strategy buffer")
        count = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=count)
        return {
            "states": self.states[indices],
            "legal_masks": self.legal_masks[indices],
            "targets": self.targets[indices],
            "weights": self.weights[indices],
        }

    def __len__(self) -> int:
        return self.size
