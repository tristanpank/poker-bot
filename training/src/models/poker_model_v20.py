"""
Poker Model V20: Pluribus-like Training Overhaul

Key improvements over V18:
1. 100% self-play training with 5 frozen snapshots.
2. Linearly-weighted opponent sampling (newer models sampled more often).
3. Pure BB profit/loss as reward (no artificial shaping).
4. Reduced action space to 7 actions (Fold, Check, Call, 33%, 66%, 100%, All-in).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import random
import copy

# 7 actions for expanded bet sizing
NUM_ACTIONS_V20 = 7
ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_33POT = 3
ACTION_RAISE_66POT = 4
ACTION_RAISE_100POT = 5
ACTION_ALL_IN = 6

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
            nn.Linear(hidden_dim // 2, NUM_ACTIONS_V20)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class OpponentPool:
    """
    Manages a pool of past model snapshots for self-play training.
    """
    
    def __init__(self, max_size: int = 20, device: str = 'cpu'):
        self.max_size = max_size
        self.device = device
        self.models = []
        self.creation_steps = []
    
    def add_snapshot(self, model: DuelingPokerNet, training_step: int):
        snapshot = copy.deepcopy(model)
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad = False
        snapshot.to(self.device)
        
        self.models.append(snapshot)
        self.creation_steps.append(max(1, training_step))
        
        if len(self.models) > self.max_size:
            self.models.pop(0)
            self.creation_steps.pop(0)
    
    def sample_opponent(self) -> Optional[DuelingPokerNet]:
        if not self.models:
            return None
            
        weights = np.array(self.creation_steps, dtype=np.float32)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
            
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.models), p=probs)
        return self.models[idx]
    
    def sample_opponents(self, n: int) -> List[Optional[DuelingPokerNet]]:
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


class PrioritizedReplayBuffer:
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
    if equity < 0.30: return 0
    elif equity < 0.45: return 1
    elif equity < 0.60: return 2
    elif equity < 0.75: return 3
    else: return 4


def compute_v20_hand_reward(profit_bb: float) -> float:
    return profit_bb
