"""
Poker Model V13: Dueling DQN with Prioritized Experience Replay

Key improvements over V12:
1. Dueling DQN architecture (separate value and advantage streams)
2. Prioritized Experience Replay for better sample efficiency
3. 6-action output for expanded bet sizing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# Action constants for V13
NUM_ACTIONS_V13 = 6
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE_SMALL = 2   # 2x min raise
ACTION_RAISE_MEDIUM = 3  # 0.5x pot
ACTION_RAISE_LARGE = 4   # 1x pot
ACTION_ALL_IN = 5


class DuelingPokerNet(nn.Module):
    """
    Dueling DQN architecture for poker.
    
    Separates value estimation V(s) from action advantages A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    
    This helps the agent learn state values independently from action selection,
    improving learning efficiency especially when many actions have similar values.
    """
    
    def __init__(self, state_dim=385, action_dim=NUM_ACTIONS_V13, hidden_dim=512):
        super().__init__()
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.action_dim = action_dim
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        # Subtracting mean advantage improves stability
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class SumTree:
    """
    Binary tree data structure for efficient priority-based sampling.
    Used by Prioritized Experience Replay.
    
    Properties:
    - Leaf nodes store priorities
    - Parent nodes store sum of children
    - Root stores total sum
    - Allows O(log n) sampling and updates
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find sample on leaf node based on cumulative sum s"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add data with given priority"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority at given tree index"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get sample based on cumulative priority s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Key features:
    - Samples transitions with probability proportional to TD-error
    - Uses importance sampling weights to correct for bias
    - Gradually increases beta (importance sampling exponent) during training
    """
    
    def __init__(self, capacity=300000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to prevent zero priorities
        self.max_priority = 1.0
    
    def _get_beta(self):
        """Linearly anneal beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, transition):
        """Add transition with max priority (new experiences are important)"""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def push_batch(self, transitions):
        """Add multiple transitions"""
        for t in transitions:
            self.push(t)
    
    def sample(self, batch_size):
        """
        Sample batch weighted by priorities.
        
        Returns:
            batch: List of transitions
            indices: Tree indices (for updating priorities)
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        
        total = self.tree.total()
        segment = total / batch_size
        
        beta = self._get_beta()
        self.frame += 1
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            if data is not 0:  # Check for valid data
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Importance sampling weights
        sampling_probs = np.array(priorities) / total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize to [0, 1]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


def compute_hand_strength_category(equity: float) -> int:
    """
    Categorize hand strength for reward shaping.
    
    Returns category 0-4:
    0: Trash (equity < 0.30)
    1: Marginal (0.30 <= equity < 0.45)
    2: Decent (0.45 <= equity < 0.60)
    3: Strong (0.60 <= equity < 0.75)
    4: Monster (equity >= 0.75)
    """
    if equity < 0.30:
        return 0
    elif equity < 0.45:
        return 1
    elif equity < 0.60:
        return 2
    elif equity < 0.75:
        return 3
    else:
        return 4


def compute_bet_sizing_efficiency(action: int, equity: float, pot_bb: float) -> float:
    """
    Evaluate bet sizing efficiency.
    
    Good sizing matches hand strength:
    - Strong hands: Larger bets (value extraction)
    - Marginal hands: Check/call or small bets (pot control)
    - Bluffs: Medium bets (fold equity vs risk)
    """
    strength = compute_hand_strength_category(equity)
    
    if action == ACTION_FOLD:
        return 0.0
    elif action == ACTION_CALL:
        # Good for marginal hands
        if strength in [1, 2]:
            return 0.1
        return 0.0
    elif action == ACTION_RAISE_SMALL:
        # Good for decent-strong hands (thin value)
        if strength in [2, 3]:
            return 0.15
        return 0.0
    elif action == ACTION_RAISE_MEDIUM:
        # Good for strong hands
        if strength == 3:
            return 0.2
        # Can be good bluff size
        if strength == 0 and pot_bb > 5:
            return 0.1
        return 0.0
    elif action == ACTION_RAISE_LARGE:
        # Good for monsters
        if strength == 4:
            return 0.25
        # Expensive bluff, small bonus if it works
        if strength == 0:
            return -0.1
        return 0.0
    elif action == ACTION_ALL_IN:
        # Only good with monsters
        if strength == 4:
            return 0.3
        # Very risky with weak hands
        if strength < 3:
            return -0.2
        return 0.0
    
    return 0.0
