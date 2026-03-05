"""
Poker Model V19: Pluribus-like Training Overhaul

Key improvements over V18:
1. 100% self-play training with 5 frozen snapshots.
2. Linearly-weighted opponent sampling (newer models sampled more often).
3. Pure BB profit/loss as reward (no artificial shaping).
4. No opponent adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import random
import copy

# 6 actions for expanded bet sizing
NUM_ACTIONS_V19 = 10
ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_33POT = 3
ACTION_RAISE_50POT = 4
ACTION_RAISE_66POT = 5
ACTION_RAISE_75POT = 6
ACTION_RAISE_100POT = 7
ACTION_RAISE_150POT = 8
ACTION_ALL_IN = 9

# Position constants (6-max)
POS_UTG = 0
POS_MP = 1
POS_CO = 2
POS_BTN = 3
POS_SB = 4
POS_BB = 5

class DuelingPokerNet(nn.Module):
    """
    Dueling DQN architecture for 6-max poker.
    Same architecture as V15/V16/V17/V18.
    """
    
    def __init__(self, state_dim: int = 520, hidden_dim: int = 512):
        super().__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS_V19)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class OpponentPool:
    """
    Manages a pool of past model snapshots for self-play training.
    Each opponent is a frozen copy of the agent from a previous training stage.
    """
    
    def __init__(self, max_size: int = 20, device: str = 'cpu'):
        self.max_size = max_size
        self.device = device
        self.models = []
        self.creation_steps = []
    
    def add_snapshot(self, model: DuelingPokerNet, training_step: int):
        """Add a model snapshot to the pool."""
        # Deep copy the model and freeze it
        snapshot = copy.deepcopy(model)
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad = False
        snapshot.to(self.device)
        
        self.models.append(snapshot)
        self.creation_steps.append(max(1, training_step))
        
        # Remove oldest if over capacity (FIFO)
        if len(self.models) > self.max_size:
            self.models.pop(0)
            self.creation_steps.pop(0)
    
    def sample_opponent(self) -> Optional[DuelingPokerNet]:
        """Linearly-weighted sampling based on training_step (Pluribus style)."""
        if not self.models:
            return None
            
        weights = np.array(self.creation_steps, dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
            
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.models), p=probs)
        return self.models[idx]
    
    def sample_opponents(self, n: int) -> List[Optional[DuelingPokerNet]]:
        """Sample n opponents with replacement, weighted by training step."""
        if not self.models:
            return [None] * n
            
        weights = np.array(self.creation_steps, dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
            
        probs = weights / weights.sum()
        indices = np.random.choice(len(self.models), size=n, p=probs, replace=True)
        return [self.models[i] for i in indices]
    
    def __len__(self):
        return len(self.models)
    
    def get_pool_info(self) -> Dict:
        """Return info about the opponent pool."""
        return {
            'size': len(self.models),
            'max_size': self.max_size,
            'steps': self.creation_steps.copy()
        }


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int = 300000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def push_batch(self, transitions):
        for t in transitions:
            self.push(t)
    
    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return [], [], []
        
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                   p=probs, replace=False)
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


def compute_hand_strength_category(equity: float) -> int:
    """Categorize hand strength based on equity."""
    if equity < 0.30:
        return 0  # Trash
    elif equity < 0.45:
        return 1  # Marginal
    elif equity < 0.60:
        return 2  # Decent
    elif equity < 0.75:
        return 3  # Strong
    else:
        return 4  # Monster


def compute_v19_hand_reward(profit_bb: float) -> float:
    """
    V19: Pure BB profit/loss.
    No artificial shaping or penalties. Just raw outcomes.
    """
    return profit_bb
