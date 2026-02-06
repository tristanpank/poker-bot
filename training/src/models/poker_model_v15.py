"""
Poker Model V15: 6-Max Table with Position-Aware Rewards

Key improvements over V14:
1. Expanded state dimension for 6-player table
2. Position-aware reward bonuses
3. Multi-way pot risk penalties
4. All V14 reward shaping functions retained
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from collections import deque
import random

# 6 actions for expanded bet sizing
NUM_ACTIONS_V15 = 6
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE_SMALL = 2
ACTION_RAISE_MEDIUM = 3
ACTION_RAISE_LARGE = 4
ACTION_ALL_IN = 5

# Position constants (6-max)
POS_UTG = 0   # Under the Gun (first to act preflop)
POS_MP = 1    # Middle Position
POS_CO = 2    # Cutoff
POS_BTN = 3   # Button (dealer, best position)
POS_SB = 4    # Small Blind
POS_BB = 5    # Big Blind


class DuelingPokerNet(nn.Module):
    """
    Dueling DQN architecture for 6-max poker.
    
    Separates value and advantage streams for better credit assignment.
    Same architecture as V14 but with larger input dimension.
    """
    
    def __init__(self, state_dim: int = 520, hidden_dim: int = 512):
        super().__init__()
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS_V15)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine using dueling formula
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
        
        # Current beta for importance sampling
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        # Sample based on priorities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                   p=probs, replace=False)
        
        # Importance sampling weights
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
    """
    Categorize hand strength based on equity.
    
    Returns:
        0: Trash (< 30%)
        1: Marginal (30-45%)
        2: Decent (45-60%)
        3: Strong (60-75%)
        4: Monster (> 75%)
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


def compute_position_bonus(position: int, won_hand: bool, profit_bb: float) -> float:
    """
    Bonus for winning from disadvantaged position.
    
    Early position is harder to play profitably, so reward
    successful play from those positions more.
    """
    if not won_hand or profit_bb <= 0:
        return 0.0
    
    # Bonus multipliers by position (early = harder = more bonus)
    position_multipliers = {
        POS_UTG: 0.25,   # Hardest position
        POS_MP: 0.15,
        POS_CO: 0.10,
        POS_BTN: 0.05,   # Easiest position
        POS_SB: 0.0,     # Forced blind
        POS_BB: 0.0,     # Forced blind
    }
    
    multiplier = position_multipliers.get(position, 0)
    # Scale bonus by profit size (capped)
    return multiplier * min(profit_bb / 10.0, 1.0)


def compute_multiway_risk_penalty(action: int, equity: float, 
                                   num_opponents_in_pot: int, risk_ratio: float) -> float:
    """
    Higher penalty for risky plays with multiple opponents.
    
    With more opponents, equity is distributed across more hands,
    making all-ins and big bets riskier.
    """
    if action != ACTION_ALL_IN:
        return 0.0
    
    strength = compute_hand_strength_category(equity)
    
    # Base penalty for all-in with different hand strengths
    if strength <= 1:  # Trash or marginal
        base_penalty = -0.6
    elif strength == 2:  # Decent
        base_penalty = -0.3
    elif strength == 3:  # Strong
        base_penalty = -0.1
    else:  # Monster
        base_penalty = 0.0
    
    # Increase penalty for each additional opponent in pot
    # More opponents = lower chance of winning
    multiway_multiplier = 1 + (num_opponents_in_pot - 1) * 0.25
    
    return base_penalty * multiway_multiplier * risk_ratio


def compute_position_consistency_bonus(position: int, action: int, equity: float) -> float:
    """
    Reward playing tighter from early position and looser from late position.
    
    This encourages proper position-aware strategy.
    """
    strength = compute_hand_strength_category(equity)
    
    # Early position (UTG, MP): reward tight play
    if position in [POS_UTG, POS_MP]:
        if action == ACTION_FOLD and strength <= 2:
            return 0.1  # Good fold from early position
        if action in [ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE] and strength <= 2:
            return -0.15  # Punish loose aggression from early position
    
    # Late position (CO, BTN): reward aggression with decent hands
    if position in [POS_CO, POS_BTN]:
        if action in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM] and strength >= 2:
            return 0.1  # Good value bet from late position
        if action == ACTION_FOLD and strength >= 3:
            return -0.1  # Missed value
    
    return 0.0


# Keep V14 reward functions for compatibility

def compute_risk_penalty(action: int, equity: float, risk_ratio: float) -> float:
    """Penalize excessive risk with weak hands."""
    strength = compute_hand_strength_category(equity)
    
    if action == ACTION_ALL_IN:
        if strength <= 1:
            return -0.5 * risk_ratio
        elif strength == 2:
            return -0.2 * risk_ratio
        elif strength == 3:
            return -0.1 * risk_ratio
    
    if action == ACTION_RAISE_LARGE:
        if strength <= 1:
            return -0.3 * risk_ratio
        elif strength == 2:
            return -0.1 * risk_ratio
    
    return 0.0


def compute_value_extraction_bonus(won_hand: bool, pot_won: float, 
                                    equity: float, pot_size_before: float) -> float:
    """Reward extracting value with strong hands."""
    if not won_hand or pot_won <= 0:
        return 0.0
    
    strength = compute_hand_strength_category(equity)
    
    if strength >= 3:
        pot_growth = pot_won / (pot_size_before + 1)
        return min(pot_growth * 0.2, 0.5)
    
    return 0.0


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
            return -0.2
    
    return 0.0


def compute_pot_building_bonus(actions_history: List[int], 
                                equities_history: List[float]) -> float:
    """Reward gradual pot building with strong hands."""
    if len(actions_history) < 2:
        return 0.0
    
    strong_streets = sum(1 for eq in equities_history if eq > 0.6)
    bet_streets = sum(1 for a in actions_history 
                      if a in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE])
    
    if strong_streets >= 2 and bet_streets >= 2:
        return 0.15 * min(strong_streets, 3)
    
    return 0.0


def compute_v15_shaped_reward(
    base_reward: float,
    action: int,
    equity: float,
    risk_ratio: float,
    pot_won: float,
    pot_size_before_action: float,
    won_hand: bool,
    actions_history: List[int],
    equities_history: List[float],
    position: int = POS_BTN,  # V15: Position
    num_opponents_in_pot: int = 1  # V15: Multi-way
) -> float:
    """
    V15 shaped reward combining all reward components.
    
    Includes V14 rewards plus:
    - Position bonus
    - Multi-way pot risk penalty
    - Position consistency bonus
    """
    # Base reward
    reward = base_reward
    
    # V14 components
    risk_penalty = compute_risk_penalty(action, equity, risk_ratio)
    value_bonus = compute_value_extraction_bonus(won_hand, pot_won, equity, pot_size_before_action)
    consistency = compute_action_consistency(action, equity)
    pot_building = compute_pot_building_bonus(actions_history, equities_history)
    
    # V15 components
    position_bonus = compute_position_bonus(position, won_hand, base_reward)
    multiway_penalty = compute_multiway_risk_penalty(action, equity, num_opponents_in_pot, risk_ratio)
    position_consistency = compute_position_consistency_bonus(position, action, equity)
    
    # Combine all components
    shaping = (risk_penalty + value_bonus + consistency + pot_building + 
               position_bonus + multiway_penalty + position_consistency)
    
    # Scale shaping by magnitude of base reward
    importance = max(abs(base_reward) / 10.0, 0.1)
    
    return reward + shaping * importance
