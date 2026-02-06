"""
Poker Model V14: Advanced Reward Shaping for Proper Poker Strategy

Key improvements over V13:
1. Risk penalty for excessive all-ins with weak hands
2. Value extraction bonus for maximizing winnings with strong hands
3. Action-hand consistency reward to match bet sizing to hand strength
4. Pot-building incentive to avoid scaring opponents off
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# Action constants for V14 (same as V13)
NUM_ACTIONS_V14 = 6
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
    
    def __init__(self, state_dim=385, action_dim=NUM_ACTIONS_V14, hidden_dim=512):
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


# ============================================================================
# V14 New Reward Functions
# ============================================================================

def compute_risk_penalty(action: int, equity: float, risk_ratio: float) -> float:
    """
    Penalize excessive risk-taking with weak hands.
    
    In real poker:
    - Going all-in risks your tournament/session life
    - Should only risk big with strong hands
    - Risk ratio = amount_risked / stack_size
    
    Args:
        action: Action taken (0-5)
        equity: Hand equity (0-1)
        risk_ratio: Fraction of stack risked (0-1)
    
    Returns:
        Penalty (negative value, or 0)
    """
    if action != ACTION_ALL_IN and action != ACTION_RAISE_LARGE:
        return 0.0
    
    strength = compute_hand_strength_category(equity)
    
    # High risk with weak hands = bad
    if action == ACTION_ALL_IN:
        if strength <= 1:  # Trash or marginal hand
            # Heavy penalty: all-in with trash is pure gambling
            return -0.5 * risk_ratio
        elif strength == 2:  # Decent hand
            # Moderate penalty: could be right but risky
            return -0.2 * risk_ratio
        elif strength == 3:  # Strong hand
            # Small penalty: aggressive but acceptable
            return -0.1 * risk_ratio
        else:  # Monster
            # No penalty, all-in with monsters is fine
            return 0.0
    
    elif action == ACTION_RAISE_LARGE:
        if strength <= 1:
            return -0.3 * risk_ratio
        elif strength == 2:
            return -0.1 * risk_ratio
        else:
            return 0.0
    
    return 0.0


def compute_value_extraction_bonus(
    action: int, 
    equity: float, 
    pot_won: float, 
    pot_size_before_action: float,
    won_hand: bool
) -> float:
    """
    Reward efficient value extraction with strong hands.
    
    The idea:
    - With strong hands, you want opponents to put money in the pot
    - Overbetting can scare them off (winning less than you could)
    - Smaller bets that get called = more value extracted
    
    Args:
        action: Action taken (0-5)
        equity: Hand equity at decision time
        pot_won: Total pot won (0 if lost)
        pot_size_before_action: Pot size when decision was made
        won_hand: Whether agent won the hand
    
    Returns:
        Bonus (positive value, or 0)
    """
    if not won_hand or pot_won <= 0:
        return 0.0
    
    strength = compute_hand_strength_category(equity)
    
    if strength >= 3:  # Strong or monster hand
        # Calculate how much "extra" value was extracted
        # If we won a big pot with a strong hand, that's good
        extraction_ratio = pot_won / (pot_size_before_action + 1e-6)
        
        # But if we went all-in and only won the blinds, that's bad
        if action == ACTION_ALL_IN and pot_won < 5:  # Won less than 5 BB
            # Opponent folded to our overbet - missed value
            return -0.3
        
        # Gradual pot building is rewarded
        if action in [ACTION_CALL, ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM]:
            if extraction_ratio > 2.0:  # Won more than 2x initial pot
                return 0.2
            elif extraction_ratio > 1.0:
                return 0.1
    
    return 0.0


def compute_action_consistency(action: int, equity: float) -> float:
    """
    Reward actions that match hand strength (proper poker strategy).
    
    Good poker:
    - Strong hands → aggressive actions (value betting)
    - Weak hands → passive actions (pot control or fold)
    - Medium hands → mixed (context dependent)
    
    Args:
        action: Action taken (0-5)
        equity: Hand equity (0-1)
    
    Returns:
        Consistency bonus/penalty
    """
    strength = compute_hand_strength_category(equity)
    
    # MONSTER (equity >= 0.75)
    if strength == 4:
        if action == ACTION_FOLD:
            return -0.5  # Terrible: folding a monster
        elif action == ACTION_CALL:
            return -0.1  # Suboptimal: should be raising
        elif action == ACTION_RAISE_MEDIUM:
            return 0.2   # Good: building pot
        elif action == ACTION_RAISE_LARGE:
            return 0.2   # Good: extracting value
        elif action == ACTION_ALL_IN:
            return 0.1   # OK but might scare off opponents
    
    # STRONG (0.60 <= equity < 0.75)
    elif strength == 3:
        if action == ACTION_FOLD:
            return -0.4
        elif action == ACTION_CALL:
            return 0.0   # Acceptable
        elif action in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM]:
            return 0.15  # Good value betting
        elif action == ACTION_RAISE_LARGE:
            return 0.1   # Aggressive but OK
        elif action == ACTION_ALL_IN:
            return -0.1  # Risky without monster
    
    # DECENT (0.45 <= equity < 0.60)
    elif strength == 2:
        if action == ACTION_FOLD:
            return -0.2  # Too tight
        elif action == ACTION_CALL:
            return 0.1   # Good pot control
        elif action == ACTION_RAISE_SMALL:
            return 0.1   # Thin value
        elif action in [ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE]:
            return 0.0   # Neutral
        elif action == ACTION_ALL_IN:
            return -0.2  # Overbetting
    
    # MARGINAL (0.30 <= equity < 0.45)
    elif strength == 1:
        if action == ACTION_FOLD:
            return 0.0   # Acceptable
        elif action == ACTION_CALL:
            return 0.05  # OK if pot odds good
        elif action == ACTION_RAISE_SMALL:
            return 0.0   # Neutral (could be bluff)
        elif action == ACTION_RAISE_MEDIUM:
            return -0.1  # Bluffing with marginal = risky
        elif action in [ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            return -0.2  # Bad bluff sizing
    
    # TRASH (equity < 0.30)
    else:  # strength == 0
        if action == ACTION_FOLD:
            return 0.15  # Good discipline
        elif action == ACTION_CALL:
            return -0.1  # Calling with trash = bad
        elif action == ACTION_RAISE_SMALL:
            return 0.0   # Could be cheap bluff, neutral
        elif action == ACTION_RAISE_MEDIUM:
            return -0.05 # Expensive bluff
        elif action in [ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            return -0.25 # Pure gambling / bad bluff
    
    return 0.0


def compute_pot_building_bonus(
    actions_taken: list, 
    equities: list, 
    pot_won: float,
    won_hand: bool
) -> float:
    """
    Reward gradual pot building with strong hands.
    
    Instead of one big bet that folds everyone out,
    multiple smaller bets that build the pot are better.
    
    Args:
        actions_taken: List of actions taken during the hand
        equities: List of equities at each decision point
        pot_won: Final pot won
        won_hand: Whether agent won
    
    Returns:
        Bonus for pot building (positive value, or 0)
    """
    if not won_hand or len(actions_taken) == 0:
        return 0.0
    
    # Count how many value bets were made with strong hands
    value_bets = 0
    for action, equity in zip(actions_taken, equities):
        strength = compute_hand_strength_category(equity)
        if strength >= 3 and action in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM]:
            value_bets += 1
    
    # Bonus if we built pot across multiple streets with strong hands
    if value_bets >= 2 and pot_won > 10:  # At least 2 value bets, won 10+ BB
        return 0.3 * min(value_bets, 3) / 3  # Max 0.3 bonus
    
    return 0.0


def compute_v14_shaped_reward(
    base_reward: float,
    action: int,
    equity: float,
    risk_ratio: float,
    pot_won: float,
    pot_size_before_action: float,
    won_hand: bool,
    actions_history: list = None,
    equities_history: list = None
) -> float:
    """
    Compute the full V14 shaped reward.
    
    Combines:
    1. Base reward (BB won/lost)
    2. Risk penalty (discourages reckless all-ins)
    3. Value extraction bonus (rewards efficient value betting)
    4. Action consistency (rewards matching bet size to hand strength)
    5. Pot building bonus (rewards gradual pot building)
    
    Args:
        base_reward: Raw BB won/lost
        action: Last action taken
        equity: Equity at last decision
        risk_ratio: Fraction of stack risked
        pot_won: Total pot won (0 if lost)
        pot_size_before_action: Pot size at last decision
        won_hand: Whether agent won
        actions_history: All actions taken (for pot building bonus)
        equities_history: All equities (for pot building bonus)
    
    Returns:
        Shaped reward value
    """
    # 1. Base reward
    shaped = base_reward
    
    # 2. Risk penalty
    risk_pen = compute_risk_penalty(action, equity, risk_ratio)
    
    # 3. Value extraction bonus
    value_bonus = compute_value_extraction_bonus(
        action, equity, pot_won, pot_size_before_action, won_hand
    )
    
    # 4. Action consistency
    consistency = compute_action_consistency(action, equity)
    
    # 5. Pot building bonus
    pot_bonus = 0.0
    if actions_history and equities_history:
        pot_bonus = compute_pot_building_bonus(
            actions_history, equities_history, pot_won, won_hand
        )
    
    # Combine all components
    # Scale bonuses/penalties relative to base reward magnitude
    base_magnitude = abs(base_reward) + 1.0  # Avoid division by zero
    shaping = (risk_pen + value_bonus + consistency + pot_bonus) * base_magnitude
    
    return shaped + shaping


# For backward compatibility with V13 imports
def compute_bet_sizing_efficiency(action: int, equity: float, pot_bb: float) -> float:
    """Legacy function for compatibility. Use compute_action_consistency instead."""
    return compute_action_consistency(action, equity)
