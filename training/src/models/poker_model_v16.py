"""
Poker Model V16: Long-Term Strategy with Session-Based Rewards

Key improvements over V15:
1. Session-based reward calculation (not single hand)
2. Severe all-in loss penalties
3. All-in frequency penalties
4. Risk-adjusted rewards
5. Bust penalties and survival bonuses
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from collections import deque
import random

# 6 actions for expanded bet sizing
NUM_ACTIONS_V16 = 6
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


class DuelingPokerNet(nn.Module):
    """
    Dueling DQN architecture for 6-max poker.
    Same architecture as V15.
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
            nn.Linear(hidden_dim // 2, NUM_ACTIONS_V16)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


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


# ============== V16 REWARD FUNCTIONS ==============

def compute_allin_outcome_penalty(action: int, won_hand: bool, equity: float) -> float:
    """
    Severe penalty for losing all-ins, scaled by how bad the decision was.
    
    Losing an all-in is always punished, but worse for marginal spots.
    Winning an all-in gets NO bonus (it's expected value, not an achievement).
    """
    if action != ACTION_ALL_IN:
        return 0.0
    
    if not won_hand:
        # Lost all-in - penalty based on equity
        if equity < 0.50:
            return -5.0  # Terrible play, gambling
        elif equity < 0.70:
            return -3.0  # Marginal spot, too risky
        elif equity < 0.85:
            return -1.5  # Okay spot, still risky
        else:
            return -0.5  # Good spot, just unlucky
    
    # Won all-in - no bonus (expected)
    return 0.0


def compute_allin_frequency_penalty(allin_count: int) -> float:
    """
    Penalize excessive all-in usage.
    
    First 2 all-ins per session are "free".
    After that, -0.5 per additional all-in.
    """
    if allin_count <= 2:
        return 0.0
    
    excess = allin_count - 2
    return -0.5 * excess


def compute_risk_adjusted_reward(base_reward: float, action: int, equity: float) -> float:
    """
    Discount high-variance plays.
    
    Variance is highest at 50% equity, lowest at 0% or 100%.
    All-in plays are discounted by variance factor.
    """
    if action != ACTION_ALL_IN:
        return base_reward
    
    # Variance = equity * (1 - equity), max 0.25 at 50%
    variance = equity * (1 - equity)
    
    # Discount factor: higher variance = lower discount
    # Range: 0.625 (at 50% equity) to 1.0 (at 0% or 100%)
    discount = 1.0 - variance * 1.5
    discount = max(0.5, min(1.0, discount))
    
    return base_reward * discount


def compute_bust_penalty(final_stack_bb: float, bust_threshold: float = 5.0) -> float:
    """
    Massive penalty for going broke.
    
    If stack falls below threshold, agent "busted" and gets harsh penalty.
    """
    if final_stack_bb < bust_threshold:
        return -20.0
    return 0.0


def compute_survival_bonus(session_completed: bool, final_stack_bb: float, 
                           starting_stack_bb: float = 100.0) -> float:
    """
    Bonus for completing a full session without busting.
    
    Rewards conservative, sustainable play.
    """
    if not session_completed:
        return 0.0
    
    # Completed session - give survival bonus
    if final_stack_bb > starting_stack_bb:
        return 2.0  # Profitable session
    elif final_stack_bb >= starting_stack_bb * 0.8:
        return 1.0  # Maintained stack
    else:
        return 0.5  # Survived but lost chips


def compute_v16_session_reward(
    session_profit_bb: float,
    final_stack_bb: float,
    starting_stack_bb: float,
    session_completed: bool,
    allin_outcomes: List[Dict],  # List of {'won': bool, 'equity': float}
) -> float:
    """
    V16 session-level reward calculation.
    
    Combines:
    - Base session profit
    - All-in outcome penalties
    - All-in frequency penalty
    - Bust penalty
    - Survival bonus
    """
    # Base reward is session profit
    reward = session_profit_bb
    
    # All-in outcome penalties
    allin_count = len(allin_outcomes)
    for outcome in allin_outcomes:
        penalty = compute_allin_outcome_penalty(
            ACTION_ALL_IN, outcome['won'], outcome['equity']
        )
        reward += penalty
    
    # All-in frequency penalty
    reward += compute_allin_frequency_penalty(allin_count)
    
    # Bust penalty
    reward += compute_bust_penalty(final_stack_bb)
    
    # Survival bonus
    reward += compute_survival_bonus(session_completed, final_stack_bb, starting_stack_bb)
    
    return reward


# Keep position-aware functions from V15

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
            return 0.1
        if action in [ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            return -0.3  # V16: Increased penalty for aggressive weak plays
    
    return 0.0


def compute_v16_hand_shaping(
    base_reward: float,
    action: int,
    equity: float,
    position: int,
    won_hand: bool,
) -> float:
    """
    Per-hand shaping for intermediate feedback.
    
    This is applied during training to each hand,
    but final session reward overrides for terminal states.
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
