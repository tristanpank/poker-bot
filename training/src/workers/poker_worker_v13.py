"""
Poker Worker V13: Realistic Stochastic Opponents and 6-Action Support

Key improvements:
1. 6 bet sizing actions (Fold, Call, SmallRaise, MedRaise, LargeRaise, AllIn)
2. 5 realistic stochastic opponent types (TAG, LAG, Rock, Calling Station, MixedBalanced)
3. Enhanced reward shaping with hand strength and bet sizing efficiency
"""

import random
from itertools import combinations
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from pokerkit import Automation, NoLimitTexasHoldem, Card, StandardHighHand, Deck
from poker_model_v13 import (
    DuelingPokerNet, NUM_ACTIONS_V13,
    ACTION_FOLD, ACTION_CALL, ACTION_RAISE_SMALL, 
    ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE, ACTION_ALL_IN,
    compute_hand_strength_category, compute_bet_sizing_efficiency
)


def flatten_cards_list(items):
    """Flatten nested card lists."""
    out = []
    if isinstance(items, Card): 
        return [items]
    for x in items:
        if isinstance(x, (list, tuple)): 
            out.extend(flatten_cards_list(x))
        else: 
            out.append(x)
    return out


def monte_carlo_equity_fast(hole_cards: List[Card], board_cards: List[Card], 
                            iterations: int = 20) -> float:
    """Fast Monte Carlo equity calculation."""
    if not hole_cards: 
        return 0.5
    
    wins = 0
    hole_cards = flatten_cards_list(hole_cards)
    board_cards = flatten_cards_list(board_cards)
    known_cards = set(hole_cards + board_cards)
    deck_cards = [c for c in Deck.STANDARD if c not in known_cards]
    needed_board = 5 - len(board_cards)
    
    for _ in range(iterations):
        random.shuffle(deck_cards)
        opp_hole = deck_cards[:2]
        sim_board = board_cards + deck_cards[2:2+needed_board]
        
        my_total = hole_cards + sim_board
        opp_total = opp_hole + sim_board
        
        my_hand = max(StandardHighHand(c) for c in combinations(my_total, 5))
        opp_hand = max(StandardHighHand(c) for c in combinations(opp_total, 5))
        
        if my_hand > opp_hand: 
            wins += 1
        elif my_hand == opp_hand: 
            wins += 0.5
    
    return wins / iterations


class StochasticOpponent:
    """
    Base class for realistic poker opponents with stochastic behavior.
    
    All opponents have randomized thresholds to avoid determinism
    and model real player tendencies.
    """
    
    def __init__(self, opp_type: str):
        self.opp_type = opp_type
        # Add random variation to thresholds each hand
        self.noise = lambda: random.gauss(0, 0.05)
    
    def select_action(self, equity: float, pot_odds: float, 
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        """Override in subclasses."""
        raise NotImplementedError


class TightAggressiveOpponent(StochasticOpponent):
    """
    TAG: Plays few hands but bets aggressively with strong holdings.
    
    Characteristics:
    - High equity threshold to play (tight)
    - Prefers raising over calling when strong (aggressive)
    - Rarely bluffs (< 10% bluff frequency)
    """
    
    def __init__(self):
        super().__init__('TAG')
        self.value_raise_threshold = 0.70  # Raise with strong
        self.call_threshold = 0.55         # Call threshold
        self.bluff_frequency = 0.08        # Low bluff rate
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        noise = self.noise()
        
        # Value betting with strong hands
        if equity > self.value_raise_threshold + noise:
            if ACTION_RAISE_LARGE in legal_actions and equity > 0.80:
                return ACTION_RAISE_LARGE
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Calling with decent hands
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        # Rare bluff
        if random.random() < self.bluff_frequency and equity < 0.35:
            if ACTION_RAISE_MEDIUM in legal_actions and pot_bb > 3:
                return ACTION_RAISE_MEDIUM
        
        # Fold weak hands
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class LooseAggressiveOpponent(StochasticOpponent):
    """
    LAG: Plays many hands aggressively with mix of value and bluffs.
    
    Characteristics:
    - Lower equity threshold (loose)
    - Frequent raising (aggressive)
    - Balanced bluff frequency (~25%)
    """
    
    def __init__(self):
        super().__init__('LAG')
        self.value_raise_threshold = 0.55
        self.call_threshold = 0.40
        self.bluff_frequency = 0.25
        self.overbet_frequency = 0.15
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        noise = self.noise()
        
        # Aggressive value betting
        if equity > self.value_raise_threshold + noise:
            # Occasional overbet with strong hands
            if equity > 0.75 and random.random() < self.overbet_frequency:
                if ACTION_ALL_IN in legal_actions:
                    return ACTION_ALL_IN
                if ACTION_RAISE_LARGE in legal_actions:
                    return ACTION_RAISE_LARGE
            
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Bluffing with weak hands
        if equity < 0.35 and random.random() < self.bluff_frequency:
            # Vary bluff sizing
            bluff_size = random.random()
            if bluff_size > 0.8 and ACTION_RAISE_LARGE in legal_actions:
                return ACTION_RAISE_LARGE
            if bluff_size > 0.4 and ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Loose calling
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class TightPassiveOpponent(StochasticOpponent):
    """
    Rock: Plays few hands, prefers calling. Only raises with monsters.
    
    Characteristics:
    - Very high equity threshold (very tight)
    - Prefers calling over raising (passive)
    - Easy to bluff but dangerous when betting
    """
    
    def __init__(self):
        super().__init__('Rock')
        self.monster_threshold = 0.85    # Only raise with monsters
        self.call_threshold = 0.60       # Need strong hand to call
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        noise = self.noise()
        
        # Only raise with monsters
        if equity > self.monster_threshold + noise:
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Call with solid hands
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        # Very tight - fold anything marginal
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class LoosePassiveOpponent(StochasticOpponent):
    """
    Calling Station: Calls too often, rarely folds, rarely raises.
    
    Characteristics:
    - Very low fold threshold (will call almost anything)
    - Rarely raises (passive)
    - Easy to value bet against, hard to bluff
    """
    
    def __init__(self):
        super().__init__('CallingStation')
        self.fold_equity_threshold = 0.20  # Only fold trash
        self.raise_threshold = 0.80        # Rarely raises
        self.calling_tendency = 0.85       # High call frequency
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        noise = self.noise()
        
        # Rare raises with strong hands
        if equity > self.raise_threshold + noise:
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Call almost everything
        if equity > self.fold_equity_threshold + noise:
            if ACTION_CALL in legal_actions:
                # Sometimes check instead of call if free
                if to_call_bb == 0 or random.random() < self.calling_tendency:
                    return ACTION_CALL
        
        # Reluctant fold only with terrible hands
        if equity < self.fold_equity_threshold and random.random() > 0.3:
            if ACTION_FOLD in legal_actions:
                return ACTION_FOLD
        
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class MixedBalancedOpponent(StochasticOpponent):
    """
    Balanced: Well-rounded player approximating GTO frequencies.
    
    Characteristics:
    - Balanced value/bluff ratios
    - Position-aware (simplified)
    - Uses pot odds properly
    - Mixes actions at key thresholds
    """
    
    def __init__(self):
        super().__init__('MixedBalanced')
        self.value_bet_ratio = 0.70  # 70% value, 30% bluff when betting
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state) -> int:
        noise = self.noise()
        
        # Decide whether to take aggressive line
        should_bet = random.random()
        
        if equity > 0.75 + noise:
            # Strong hand - value bet
            if should_bet < 0.85:  # 85% bet frequency
                size_roll = random.random()
                if size_roll > 0.7 and ACTION_RAISE_LARGE in legal_actions:
                    return ACTION_RAISE_LARGE
                if size_roll > 0.3 and ACTION_RAISE_MEDIUM in legal_actions:
                    return ACTION_RAISE_MEDIUM
                if ACTION_RAISE_SMALL in legal_actions:
                    return ACTION_RAISE_SMALL
            return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity > 0.55 + noise:
            # Medium strong - mixed strategy
            if should_bet < 0.4:  # 40% bet frequency
                if ACTION_RAISE_SMALL in legal_actions:
                    return ACTION_RAISE_SMALL
            
            # Check equity vs pot odds for call
            if equity > pot_odds + 0.05:
                return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity > 0.40 + noise:
            # Marginal - mostly call/check if odds are good
            if equity > pot_odds:
                return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity < 0.30:
            # Weak - bluff sometimes
            bluff_roll = random.random()
            if bluff_roll < 0.20:  # 20% bluff frequency with weak hands
                if ACTION_RAISE_MEDIUM in legal_actions:
                    return ACTION_RAISE_MEDIUM
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


def create_opponent(opp_type: str) -> StochasticOpponent:
    """Factory function to create opponent by type."""
    opponents = {
        'TAG': TightAggressiveOpponent,
        'LAG': LooseAggressiveOpponent,
        'Rock': TightPassiveOpponent,
        'CallingStation': LoosePassiveOpponent,
        'MixedBalanced': MixedBalancedOpponent,
    }
    return opponents.get(opp_type, MixedBalancedOpponent)()


def run_single_episode_v13(args):
    """
    Run a single poker episode with V13 features.
    
    Returns all transitions and stats.
    This function runs in a separate process.
    """
    episode_seed, epsilon, opp_type, equity_iterations, model_state_dict = args
    random.seed(episode_seed)
    
    # Load model for action selection (385 = 380 base + 5 hand strength categories)
    model = DuelingPokerNet(state_dim=385)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create stochastic opponent
    opponent = create_opponent(opp_type)
    
    # Game setup
    starting_stack = 1000
    big_blind = 10
    small_blind = 5
    
    state = NoLimitTexasHoldem.create_state(
        automations=(
            Automation.ANTE_POSTING, Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING, Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING, Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
        ),
        ante_trimming_status=True, raw_antes={-1: 0},
        raw_blinds_or_straddles=(small_blind, big_blind),
        min_bet=big_blind,
        raw_starting_stacks=[starting_stack, starting_stack],
        player_count=2,
    )
    
    while state.can_deal_hole():
        state.deal_hole()
    
    def run_automations():
        while state.can_burn_card(): state.burn_card('??')
        while state.can_deal_board(): state.deal_board()
        while state.can_push_chips(): state.push_chips()
        while state.can_pull_chips(): state.pull_chips()
    
    run_automations()
    
    def get_observation():
        """Build state observation vector."""
        state_vector = []
        hole = flatten_cards_list(state.hole_cards[0])
        board = flatten_cards_list(state.board_cards)
        
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        
        # Encode hole cards (2 x 52)
        for i in range(2):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(hole):
                idx = ranks.index(hole[i].rank) * 4 + suits.index(hole[i].suit)
                encoding[idx] = 1.0
            state_vector.extend(encoding)
        
        # Encode board cards (5 x 52)
        for i in range(5):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(board):
                idx = ranks.index(board[i].rank) * 4 + suits.index(board[i].suit)
                encoding[idx] = 1.0
            state_vector.extend(encoding)
        
        # Stack and pot info
        total_pot = sum(state.bets)
        current_bet = max(state.bets)
        my_bet = state.bets[0]
        to_call = current_bet - my_bet
        my_stack = state.stacks[0]
        opp_stack = state.stacks[1]
        
        # Equity calculation
        equity = monte_carlo_equity_fast(hole, board, equity_iterations)
        state_vector.append(equity)
        
        # Pot odds and related
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(pot_odds)
        state_vector.append(min(to_call / (total_pot + 1e-6), 2.0))
        state_vector.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
        state_vector.append(min(my_stack, opp_stack) / starting_stack)
        state_vector.append(my_stack / starting_stack)
        state_vector.append(opp_stack / starting_stack)
        state_vector.append(total_pot / (starting_stack * 2))
        
        # EV features
        breakeven_equity = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(breakeven_equity)
        excess_equity = equity - breakeven_equity
        state_vector.append(excess_equity)
        state_vector.append((starting_stack - my_stack) / starting_stack)
        state_vector.append(1.0 if to_call > 0 else 0.0)
        
        # Hand strength category
        strength_cat = compute_hand_strength_category(equity)
        for i in range(5):
            state_vector.append(1.0 if i == strength_cat else 0.0)
        
        # Street encoding
        street = [0.0, 0.0, 0.0, 0.0]
        if len(board) == 0: street[0] = 1.0
        elif len(board) == 3: street[1] = 1.0
        elif len(board) == 4: street[2] = 1.0
        else: street[3] = 1.0
        state_vector.extend(street)
        
        return np.array(state_vector, dtype=np.float32), {
            'equity': equity, 
            'pot_bb': total_pot / big_blind,
            'to_call_bb': to_call / big_blind, 
            'breakeven_equity': breakeven_equity,
            'excess_equity': excess_equity,
            'hand_strength': strength_cat
        }
    
    def get_legal_actions():
        """Get list of legal action indices for 6-action space."""
        legal = []
        if state.can_fold(): 
            legal.append(ACTION_FOLD)
        if state.can_check_or_call(): 
            legal.append(ACTION_CALL)
        if state.can_complete_bet_or_raise_to():
            # All raise actions are legal when raising is allowed
            legal.append(ACTION_RAISE_SMALL)
            legal.append(ACTION_RAISE_MEDIUM)
            legal.append(ACTION_RAISE_LARGE)
            legal.append(ACTION_ALL_IN)
        return legal if legal else [ACTION_CALL]
    
    def execute_action(action: int, actor: int = 0):
        """Execute action with proper bet sizing."""
        if action == ACTION_FOLD:
            if state.can_fold(): 
                state.fold()
            elif state.can_check_or_call(): 
                state.check_or_call()
        
        elif action == ACTION_CALL:
            if state.can_check_or_call(): 
                state.check_or_call()
            elif state.can_fold(): 
                state.fold()
        
        elif action in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, 
                        ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            if state.can_complete_bet_or_raise_to():
                min_r = state.min_completion_betting_or_raising_to_amount
                max_r = state.max_completion_betting_or_raising_to_amount
                pot = sum(state.bets)
                
                if action == ACTION_RAISE_SMALL:
                    # 2x min raise
                    amount = min(min_r * 2, max_r)
                elif action == ACTION_RAISE_MEDIUM:
                    # 0.5x pot
                    amount = min(max(min_r, pot * 0.5), max_r)
                elif action == ACTION_RAISE_LARGE:
                    # 1x pot
                    amount = min(max(min_r, pot), max_r)
                elif action == ACTION_ALL_IN:
                    amount = max_r
                
                state.complete_bet_or_raise_to(int(amount))
            elif state.can_check_or_call(): 
                state.check_or_call()
    
    def opponent_action_v13(legal: List[int]) -> int:
        """Get opponent action using stochastic opponent."""
        opp_hole = flatten_cards_list(state.hole_cards[1])
        board = flatten_cards_list(state.board_cards)
        equity = monte_carlo_equity_fast(opp_hole, board, equity_iterations)
        
        total_pot = sum(state.bets)
        current_bet = max(state.bets)
        opp_bet = state.bets[1]
        to_call = current_bet - opp_bet
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        
        return opponent.select_action(
            equity=equity,
            pot_odds=pot_odds,
            to_call_bb=to_call / big_blind,
            pot_bb=total_pot / big_blind,
            legal_actions=legal,
            state=state
        )
    
    # Play episode
    transitions = []
    contexts = []
    actions_taken = []
    action_counts = {i: 0 for i in range(NUM_ACTIONS_V13)}
    
    pending_obs = None
    pending_action = None
    pending_context = None
    
    while state.status is not False:
        if state.actor_index == 0:  # Agent
            obs, ctx = get_observation()
            legal = get_legal_actions()
            
            if pending_obs is not None:
                transitions.append((pending_obs, pending_action, 0.0, obs, False))
                if pending_context:
                    contexts.append(pending_context)
                    actions_taken.append(pending_action)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(legal)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = model(state_tensor).squeeze(0)
                    # Mask illegal actions
                    masked_q = torch.full_like(q_values, float('-inf'))
                    for a in legal:
                        masked_q[a] = q_values[a]
                    action = masked_q.argmax().item()
            
            action_counts[action] += 1
            pending_obs = obs
            pending_action = action
            pending_context = ctx
            
            execute_action(action, actor=0)
            run_automations()
        
        else:  # Opponent
            legal = get_legal_actions()
            action = opponent_action_v13(legal)
            execute_action(action, actor=1)
            run_automations()
    
    # Terminal
    final_reward = (state.stacks[0] - starting_stack) / big_blind
    if pending_obs is not None:
        term_obs, _ = get_observation()
        transitions.append((pending_obs, pending_action, 0.0, term_obs, True))
        if pending_context:
            contexts.append(pending_context)
            actions_taken.append(pending_action)
    
    return {
        'transitions': transitions,
        'contexts': contexts,
        'actions': actions_taken,
        'final_reward': final_reward,
        'action_counts': action_counts,
        'opp_type': opp_type,
        'won': final_reward > 0
    }
