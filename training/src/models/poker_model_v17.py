"""
Poker Model V17: Self-Play Training with Stronger Penalties

Key improvements over V16:
1. Much stronger all-in loss penalties (-10 BB base vs -3 BB)
2. Harsher bust penalty (-50 BB vs -20 BB)
3. Lower all-in frequency threshold (8% vs 15%)
4. Self-play opponent support
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import random
import copy

# 6 actions for expanded bet sizing
NUM_ACTIONS_V17 = 6
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE_SMALL = 2
ACTION_RAISE_MEDIUM = 3
ACTION_RAISE_LARGE = 4
ACTION_ALL_IN = 5

# Position constants (6-max)
POS_UTG = 0
POS_MP = 1
POS_CO = 2
POS_BTN = 3
POS_SB = 4
POS_BB = 5

# V17 PENALTY PARAMETERS (much stronger than V16)
ALLIN_LOSS_PENALTY_BASE = -10.0  # V16 was -5.0 for worst case
ALLIN_LOSS_MARGINAL = -6.0      # V16 was -3.0
ALLIN_LOSS_OKAY = -3.0          # V16 was -1.5
ALLIN_LOSS_UNLUCKY = -1.0       # V16 was -0.5
BUST_PENALTY = -50.0            # V16 was -20.0
ALLIN_FREQUENCY_FREE = 2        # Free all-ins per 30 hands
ALLIN_FREQUENCY_PENALTY = -1.0  # V16 was -0.5


class DuelingPokerNet(nn.Module):
    """
    Dueling DQN architecture for 6-max poker.
    Same architecture as V15/V16.
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
            nn.Linear(hidden_dim // 2, NUM_ACTIONS_V17)
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
        self.creation_steps.append(training_step)
        
        # Remove oldest if over capacity (FIFO)
        if len(self.models) > self.max_size:
            self.models.pop(0)
            self.creation_steps.pop(0)
    
    def sample_opponent(self) -> Optional[DuelingPokerNet]:
        """Randomly sample an opponent from the pool."""
        if not self.models:
            return None
        return random.choice(self.models)
    
    def sample_opponents(self, n: int) -> List[Optional[DuelingPokerNet]]:
        """Sample n opponents (with replacement if pool is smaller)."""
        if not self.models:
            return [None] * n
        return [random.choice(self.models) for _ in range(n)]
    
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


# ============== V17 REWARD FUNCTIONS (STRONGER PENALTIES) ==============

def compute_allin_outcome_penalty(action: int, won_hand: bool, equity: float) -> float:
    """
    Severe penalty for losing all-ins, scaled by how bad the decision was.
    V17: Much stronger penalties than V16.
    """
    if action != ACTION_ALL_IN:
        return 0.0
    
    if not won_hand:
        # Lost all-in - penalty based on equity (V17: 2x V16 values)
        if equity < 0.50:
            return ALLIN_LOSS_PENALTY_BASE  # -10.0: Terrible play, gambling
        elif equity < 0.70:
            return ALLIN_LOSS_MARGINAL      # -6.0: Marginal spot, too risky
        elif equity < 0.85:
            return ALLIN_LOSS_OKAY          # -3.0: Okay spot, still risky
        else:
            return ALLIN_LOSS_UNLUCKY       # -1.0: Good spot, just unlucky
    
    # Won all-in - no bonus (expected)
    return 0.0


def compute_allin_frequency_penalty(allin_count: int, session_hands: int = 30) -> float:
    """
    Penalize excessive all-in usage.
    V17: Target <8% all-in rate (was 15% in V16)
    """
    if allin_count <= ALLIN_FREQUENCY_FREE:
        return 0.0
    
    excess = allin_count - ALLIN_FREQUENCY_FREE
    return ALLIN_FREQUENCY_PENALTY * excess  # -1.0 per excess (was -0.5)


def compute_risk_adjusted_reward(base_reward: float, action: int, equity: float) -> float:
    """
    Discount high-variance plays.
    """
    if action != ACTION_ALL_IN:
        return base_reward
    
    variance = equity * (1 - equity)
    discount = 1.0 - variance * 1.5
    discount = max(0.5, min(1.0, discount))
    
    return base_reward * discount


def compute_bust_penalty(final_stack_bb: float, bust_threshold: float = 5.0) -> float:
    """
    Massive penalty for going broke.
    V17: Much harsher (-50 BB vs -20 BB in V16)
    """
    if final_stack_bb < bust_threshold:
        return BUST_PENALTY
    return 0.0


def compute_survival_bonus(session_completed: bool, final_stack_bb: float, 
                           starting_stack_bb: float = 100.0) -> float:
    """
    Bonus for completing a full session without busting.
    """
    if not session_completed:
        return 0.0
    
    if final_stack_bb > starting_stack_bb:
        return 3.0  # Profitable session (V17: increased from 2.0)
    elif final_stack_bb >= starting_stack_bb * 0.8:
        return 1.5  # Maintained stack (V17: increased from 1.0)
    else:
        return 0.5  # Survived but lost chips


def compute_v17_session_reward(
    session_profit_bb: float,
    final_stack_bb: float,
    starting_stack_bb: float,
    session_completed: bool,
    allin_outcomes: List[Dict],  # List of {'won': bool, 'equity': float}
    session_hands: int = 30,
) -> float:
    """
    V17 session-level reward calculation.
    Same structure as V16 but with stronger penalties.
    """
    # Base reward is session profit
    reward = session_profit_bb
    
    # All-in outcome penalties (V17: stronger)
    allin_count = len(allin_outcomes)
    for outcome in allin_outcomes:
        penalty = compute_allin_outcome_penalty(
            ACTION_ALL_IN, outcome['won'], outcome['equity']
        )
        reward += penalty
    
    # All-in frequency penalty (V17: stronger)
    reward += compute_allin_frequency_penalty(allin_count, session_hands)
    
    # Bust penalty (V17: -50 BB)
    reward += compute_bust_penalty(final_stack_bb)
    
    # Survival bonus (V17: slightly higher)
    reward += compute_survival_bonus(session_completed, final_stack_bb, starting_stack_bb)
    
    return reward


# Keep position-aware functions from V15/V16

def compute_position_bonus(position: int, won_hand: bool, profit_bb: float) -> float:
    """Bonus for winning from disadvantaged position."""
    if not won_hand or profit_bb <= 0:
        return 0.0
    
    position_multipliers = {
        POS_UTG: 0.25,
        POS_MP: 0.15,
        POS_CO: 0.10,
        POS_BTN: 0.05,
        POS_SB: 0.0,
        POS_BB: 0.0,
    }
    
    multiplier = position_multipliers.get(position, 0)
    return multiplier * min(profit_bb / 10.0, 1.0)


def compute_action_consistency(action: int, equity: float) -> float:
    """Reward actions that match hand strength."""
    strength = compute_hand_strength_category(equity)
    
    if strength >= 3:
        if action in [ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE]:
            return 0.15
        if action == ACTION_FOLD:
            return -0.2
    
    if strength <= 1:
        if action == ACTION_FOLD:
            return 0.15  # V17: increased fold reward for weak hands
        if action in [ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            return -0.5  # V17: Increased penalty for aggressive weak plays
    
    return 0.0


def compute_v17_hand_shaping(
    base_reward: float,
    action: int,
    equity: float,
    position: int,
    won_hand: bool,
) -> float:
    """
    Per-hand shaping for intermediate feedback.
    """
    shaped = base_reward
    
    # Position bonus
    shaped += compute_position_bonus(position, won_hand, base_reward)
    
    # Action consistency (but NOT all-in penalty here - that's at session level)
    if action != ACTION_ALL_IN:
        shaped += compute_action_consistency(action, equity)
    
    # Risk adjustment for all-ins
    if action == ACTION_ALL_IN:
        shaped = compute_risk_adjusted_reward(shaped, action, equity)
    
    return shaped
