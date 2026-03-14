import copy
import os
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from poker_state_v24 import ACTION_COUNT_V24, PUBLIC_BELIEF_STATE_DIM_V24, STATE_DIM_V24


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _BasePokerNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        head_name: str,
        init_weights: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.input_layer = nn.Linear(self.state_dim, self.hidden_dim)
        self.input_activation = nn.GELU()
        self.block1 = MLPBlock(self.hidden_dim)
        self.block2 = MLPBlock(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.action_dim),
        )
        setattr(self, head_name, head)
        self._head_name = head_name
        if init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_activation(self.input_layer(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.output_norm(x)

    def _head(self) -> nn.Sequential:
        return getattr(self, self._head_name)

    def clone_cpu(self):
        clone = self.__class__(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            init_weights=False,
        )
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        clone.eval()
        clone.to("cpu")
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone


class PokerDeepCFRNet(_BasePokerNet):
    def __init__(
        self,
        state_dim: int = STATE_DIM_V24,
        hidden_dim: int = 160,
        action_dim: int = ACTION_COUNT_V24,
        init_weights: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            head_name="strategy_head",
            init_weights=init_weights,
        )

    @property
    def policy_head(self) -> nn.Sequential:
        return self.strategy_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"strategy": self.forward_strategy(x)}

    def forward_strategy(self, x: torch.Tensor) -> torch.Tensor:
        return self.strategy_head(self._forward_trunk(x))

    def forward_regret(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_strategy(x)

    def forward_postflop_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_strategy(x)

    def forward_exploit(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_strategy(x)


class AdvantageNet(_BasePokerNet):
    def __init__(
        self,
        state_dim: int = STATE_DIM_V24,
        hidden_dim: int = 160,
        action_dim: int = ACTION_COUNT_V24,
        init_weights: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            head_name="regret_head",
            init_weights=init_weights,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"regret": self.forward_regret(x)}

    def forward_regret(self, x: torch.Tensor) -> torch.Tensor:
        return self.regret_head(self._forward_trunk(x))


class PokerLeafValueNet(nn.Module):
    def __init__(
        self,
        state_dim: int = PUBLIC_BELIEF_STATE_DIM_V24,
        hidden_dim: int = 96,
        init_weights: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_layer = nn.Linear(self.state_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.block1 = MLPBlock(self.hidden_dim)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        if init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        return self.block1(x)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._forward_trunk(x)).squeeze(-1)

    def clone_cpu(self) -> "PokerLeafValueNet":
        clone = PokerLeafValueNet(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            init_weights=False,
        )
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        clone.eval()
        clone.to("cpu")
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone


def load_compatible_state_dict(model: PokerDeepCFRNet, state_dict: Dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("regret_head")]
    missing = [key for key in incompatible.missing_keys if not key.startswith("regret_head")]
    if missing or unexpected:
        raise RuntimeError(
            "Incompatible checkpoint for PokerDeepCFRNet "
            f"(missing: {', '.join(missing) if missing else '-'}; "
            f"unexpected: {', '.join(unexpected) if unexpected else '-'})"
        )


def load_compatible_advantage_state_dict(model: AdvantageNet, state_dict: Dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("strategy_head")]
    missing = [key for key in incompatible.missing_keys if not key.startswith("strategy_head")]
    if missing or unexpected:
        raise RuntimeError(
            "Incompatible checkpoint for AdvantageNet "
            f"(missing: {', '.join(missing) if missing else '-'}; "
            f"unexpected: {', '.join(unexpected) if unexpected else '-'})"
        )


def load_compatible_leaf_value_state_dict(model: PokerLeafValueNet, state_dict: Dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible checkpoint for PokerLeafValueNet "
            f"(missing: {', '.join(incompatible.missing_keys) if incompatible.missing_keys else '-'}; "
            f"unexpected: {', '.join(incompatible.unexpected_keys) if incompatible.unexpected_keys else '-'})"
        )


def _normalize_legal_mask(legal_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if mask.shape[0] != ACTION_COUNT_V24:
        raise ValueError(f"Expected legal mask of length {ACTION_COUNT_V24}, got {mask.shape}")
    mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
    if float(mask.sum()) <= 0.0:
        mask[:] = 1.0
    return mask


def _default_safe_policy(mask: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(mask, dtype=np.float32)
    if mask.shape[0] > 1 and mask[1] > 0.5:
        probs[1] = 1.0
    elif mask.shape[0] > 2 and mask[2] > 0.5:
        probs[2] = 0.7
        if mask.shape[0] > 0 and mask[0] > 0.5:
            probs[0] = 0.3
        else:
            probs[2] = 1.0
    elif mask.shape[0] > 0 and mask[0] > 0.5:
        probs[0] = 1.0
    else:
        probs = mask.copy()
    total = float(probs.sum())
    if total <= 1e-8:
        return (mask / float(mask.sum())).astype(np.float32)
    return (probs / total).astype(np.float32)


def regret_matching(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    fallback_policy: Optional[np.ndarray] = None,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.shape[0] != ACTION_COUNT_V24:
        raise ValueError(f"Expected logits of length {ACTION_COUNT_V24}, got {logits.shape}")
    mask = _normalize_legal_mask(legal_mask)
    regrets = np.maximum(logits, 0.0) * mask
    total = float(regrets.sum())
    if total <= 1e-8:
        if fallback_policy is None:
            return _default_safe_policy(mask)
        fallback = np.asarray(fallback_policy, dtype=np.float32).reshape(-1)
        fallback = fallback * mask
        denom = float(fallback.sum())
        if denom <= 1e-8:
            return _default_safe_policy(mask)
        return (fallback / denom).astype(np.float32)
    return (regrets / total).astype(np.float32)


def masked_policy(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.shape[0] != ACTION_COUNT_V24:
        raise ValueError(f"Expected logits of length {ACTION_COUNT_V24}, got {logits.shape}")
    mask = _normalize_legal_mask(legal_mask)
    masked = np.full_like(logits, -1e9, dtype=np.float32)
    legal = mask > 0.5
    masked[legal] = logits[legal]
    max_val = float(masked.max())
    exp_vals = np.zeros_like(masked, dtype=np.float32)
    exp_vals[legal] = np.exp(np.clip(masked[legal] - max_val, -20.0, 0.0))
    denom = float(exp_vals.sum())
    if denom <= 1e-8:
        return _default_safe_policy(mask)
    return (exp_vals / denom).astype(np.float32)


class AdvantageBuffer:
    def __init__(
        self,
        capacity: int = 131_072,
        state_dim: int = STATE_DIM_V24,
        action_dim: int = ACTION_COUNT_V24,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.legal_masks = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.targets = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.weights = np.ones(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def clear(self) -> None:
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
        return int(self.size)


class StrategyBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = STATE_DIM_V24,
        action_dim: int = ACTION_COUNT_V24,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.legal_masks = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.targets = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
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
        return int(self.size)
