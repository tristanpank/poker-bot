"""
Poker Worker V15: 6-Max Table Training

Key improvements over V14:
1. 6-player table instead of heads-up
2. Agent at random position each hand
3. Expanded observation space for multiple opponents
4. Multi-opponent equity calculation
5. Position-aware context tracking
"""

import random
from itertools import combinations
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from pokerkit import Automation, NoLimitTexasHoldem, Card, StandardHighHand, Deck
from poker_model_v15 import (
    DuelingPokerNet, NUM_ACTIONS_V15,
    ACTION_FOLD, ACTION_CALL, ACTION_RAISE_SMALL, 
    ACTION_RAISE_MEDIUM, ACTION_RAISE_LARGE, ACTION_ALL_IN,
    POS_UTG, POS_MP, POS_CO, POS_BTN, POS_SB, POS_BB,
    compute_hand_strength_category, compute_v15_shaped_reward
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


def monte_carlo_equity_multiway(hole_cards: List[Card], board_cards: List[Card], 
                                 num_opponents: int = 1, iterations: int = 20) -> float:
    """
    Monte Carlo equity calculation against multiple opponents.
    
    With more opponents, equity naturally decreases as the pot is 
    distributed across more potential winners.
    """
    if not hole_cards: 
        return 0.5
    
    wins = 0
    hole_cards = flatten_cards_list(hole_cards)
    board_cards = flatten_cards_list(board_cards)
    known_cards = set(hole_cards + board_cards)
    deck_cards = [c for c in Deck.STANDARD if c not in known_cards]
    needed_board = 5 - len(board_cards)
    cards_per_opponent = 2
    
    for _ in range(iterations):
        random.shuffle(deck_cards)
        
        # Deal cards to opponents
        idx = 0
        opponent_hands = []
        for _ in range(num_opponents):
            opp_hole = deck_cards[idx:idx + cards_per_opponent]
            opponent_hands.append(opp_hole)
            idx += cards_per_opponent
        
        # Complete board
        sim_board = board_cards + deck_cards[idx:idx + needed_board]
        
        # Calculate hands
        my_total = hole_cards + sim_board
        my_hand = max(StandardHighHand(c) for c in combinations(my_total, 5))
        
        # Compare against all opponents
        i_win = True
        ties = 0
        for opp_hole in opponent_hands:
            opp_total = opp_hole + sim_board
            opp_hand = max(StandardHighHand(c) for c in combinations(opp_total, 5))
            if opp_hand > my_hand:
                i_win = False
                break
            elif opp_hand == my_hand:
                ties += 1
        
        if i_win:
            if ties > 0:
                wins += 1.0 / (ties + 1)  # Split pot
            else:
                wins += 1
    
    return wins / iterations


class StochasticOpponent:
    """Base class for realistic poker opponents."""
    
    def __init__(self, opp_type: str):
        self.opp_type = opp_type
        self.noise = lambda: random.gauss(0, 0.05)
    
    def select_action(self, equity: float, pot_odds: float, 
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state, 
                      num_opponents: int = 1) -> int:
        raise NotImplementedError


class TightAggressiveOpponent(StochasticOpponent):
    """TAG: Plays few hands but bets aggressively with strong holdings."""
    
    def __init__(self):
        super().__init__('TAG')
        self.value_raise_threshold = 0.70
        self.call_threshold = 0.55
        self.bluff_frequency = 0.08
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state,
                      num_opponents: int = 1) -> int:
        noise = self.noise()
        # Tighten up with more opponents
        adjustment = num_opponents * 0.02
        
        if equity > self.value_raise_threshold + noise + adjustment:
            if ACTION_RAISE_LARGE in legal_actions and equity > 0.80:
                return ACTION_RAISE_LARGE
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        if random.random() < self.bluff_frequency / (num_opponents + 1) and equity < 0.35:
            if ACTION_RAISE_MEDIUM in legal_actions and pot_bb > 3:
                return ACTION_RAISE_MEDIUM
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class LooseAggressiveOpponent(StochasticOpponent):
    """LAG: Plays many hands aggressively with mix of value and bluffs."""
    
    def __init__(self):
        super().__init__('LAG')
        self.value_raise_threshold = 0.55
        self.call_threshold = 0.40
        self.bluff_frequency = 0.25
        self.overbet_frequency = 0.15
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state,
                      num_opponents: int = 1) -> int:
        noise = self.noise()
        # Slightly tighter with more opponents but still aggressive
        adjustment = num_opponents * 0.01
        
        if equity > self.value_raise_threshold + noise + adjustment:
            if equity > 0.75 and random.random() < self.overbet_frequency:
                if ACTION_ALL_IN in legal_actions:
                    return ACTION_ALL_IN
                if ACTION_RAISE_LARGE in legal_actions:
                    return ACTION_RAISE_LARGE
            
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        # Less bluffing into multiple opponents
        if equity < 0.35 and random.random() < self.bluff_frequency / num_opponents:
            bluff_size = random.random()
            if bluff_size > 0.8 and ACTION_RAISE_LARGE in legal_actions:
                return ACTION_RAISE_LARGE
            if bluff_size > 0.4 and ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class TightPassiveOpponent(StochasticOpponent):
    """Rock: Plays few hands, prefers calling. Only raises with monsters."""
    
    def __init__(self):
        super().__init__('Rock')
        self.monster_threshold = 0.85
        self.call_threshold = 0.60
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state,
                      num_opponents: int = 1) -> int:
        noise = self.noise()
        # Even tighter with more opponents
        adjustment = num_opponents * 0.03
        
        if equity > self.monster_threshold + noise:
            if ACTION_RAISE_MEDIUM in legal_actions:
                return ACTION_RAISE_MEDIUM
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions:
                return ACTION_CALL
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class LoosePassiveOpponent(StochasticOpponent):
    """Calling Station: Calls too often, rarely folds, rarely raises."""
    
    def __init__(self):
        super().__init__('CallingStation')
        self.fold_equity_threshold = 0.20
        self.raise_threshold = 0.80
        self.calling_tendency = 0.85
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state,
                      num_opponents: int = 1) -> int:
        noise = self.noise()
        
        if equity > self.raise_threshold + noise:
            if ACTION_RAISE_SMALL in legal_actions:
                return ACTION_RAISE_SMALL
        
        if equity > self.fold_equity_threshold + noise:
            if ACTION_CALL in legal_actions:
                if to_call_bb == 0 or random.random() < self.calling_tendency:
                    return ACTION_CALL
        
        if equity < self.fold_equity_threshold and random.random() > 0.3:
            if ACTION_FOLD in legal_actions:
                return ACTION_FOLD
        
        return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]


class MixedBalancedOpponent(StochasticOpponent):
    """Balanced: Well-rounded player approximating GTO frequencies."""
    
    def __init__(self):
        super().__init__('MixedBalanced')
        self.value_bet_ratio = 0.70
    
    def select_action(self, equity: float, pot_odds: float,
                      to_call_bb: float, pot_bb: float,
                      legal_actions: List[int], state,
                      num_opponents: int = 1) -> int:
        noise = self.noise()
        should_bet = random.random()
        # Adjust for multiway
        adjustment = num_opponents * 0.02
        
        if equity > 0.75 + noise:
            if should_bet < 0.85:
                size_roll = random.random()
                if size_roll > 0.7 and ACTION_RAISE_LARGE in legal_actions:
                    return ACTION_RAISE_LARGE
                if size_roll > 0.3 and ACTION_RAISE_MEDIUM in legal_actions:
                    return ACTION_RAISE_MEDIUM
                if ACTION_RAISE_SMALL in legal_actions:
                    return ACTION_RAISE_SMALL
            return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity > 0.55 + noise + adjustment:
            if should_bet < 0.4:
                if ACTION_RAISE_SMALL in legal_actions:
                    return ACTION_RAISE_SMALL
            
            if equity > pot_odds + 0.05:
                return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity > 0.40 + noise + adjustment:
            if equity > pot_odds:
                return ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0]
        
        elif equity < 0.30:
            # Less bluffing into multiple opponents
            bluff_roll = random.random()
            if bluff_roll < 0.20 / num_opponents:
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


# Position names for logging
POSITION_NAMES = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']


def run_single_episode_v15(args):
    """
    Run a single 6-max poker episode.
    
    Key features:
    - 6 players: 1 agent + 5 opponents (one of each type)
    - Agent at random position each hand
    - Extended observation space with position features
    - Multi-opponent equity calculation
    """
    episode_seed, epsilon, equity_iterations, model_state_dict = args
    random.seed(episode_seed)
    
    # Load model (520 = expanded state for 6-max)
    model = DuelingPokerNet(state_dim=520)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create all 5 opponent types
    opponent_types = ['TAG', 'LAG', 'Rock', 'CallingStation', 'MixedBalanced']
    opponents = [create_opponent(t) for t in opponent_types]
    random.shuffle(opponents)  # Randomize seating
    
    # Agent position (0-5, random each hand)
    agent_position = random.randint(0, 5)
    
    # Game setup
    num_players = 6
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
        raw_starting_stacks=[starting_stack] * num_players,
        player_count=num_players,
    )
    
    while state.can_deal_hole():
        state.deal_hole()
    
    def run_automations():
        while state.can_burn_card(): state.burn_card('??')
        while state.can_deal_board(): state.deal_board()
        while state.can_push_chips(): state.push_chips()
        while state.can_pull_chips(): state.pull_chips()
    
    run_automations()
    
    # Track which players are still in the hand
    players_in_hand = [True] * num_players
    
    def count_active_opponents():
        """Count opponents still in hand."""
        return sum(1 for i, active in enumerate(players_in_hand) 
                   if active and i != agent_position)
    
    def get_observation():
        """Build extended observation vector for 6-max."""
        state_vector = []
        hole = flatten_cards_list(state.hole_cards[agent_position])
        board = flatten_cards_list(state.board_cards)
        
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        
        # Encode hole cards (2 x 52 = 104)
        for i in range(2):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(hole):
                idx = ranks.index(hole[i].rank) * 4 + suits.index(hole[i].suit)
                encoding[idx] = 1.0
            state_vector.extend(encoding)
        
        # Encode board cards (5 x 52 = 260)
        for i in range(5):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(board):
                idx = ranks.index(board[i].rank) * 4 + suits.index(board[i].suit)
                encoding[idx] = 1.0
            state_vector.extend(encoding)
        
        # Stack and pot info
        total_pot = sum(state.bets)
        current_bet = max(state.bets)
        my_bet = state.bets[agent_position]
        to_call = current_bet - my_bet
        my_stack = state.stacks[agent_position]
        
        # Get all opponent stacks
        opp_stacks = [state.stacks[i] for i in range(num_players) if i != agent_position]
        
        # Number of active opponents
        active_opponents = count_active_opponents()
        
        # Calculate equity against active opponents
        equity = monte_carlo_equity_multiway(hole, board, active_opponents, equity_iterations)
        state_vector.append(equity)
        
        # Pot odds
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(pot_odds)
        state_vector.append(min(to_call / (total_pot + 1e-6), 2.0))
        
        # Stack ratios
        state_vector.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
        state_vector.append(my_stack / starting_stack)
        state_vector.append(total_pot / (starting_stack * num_players))
        
        # Opponent stacks (normalized, 5 opponents)
        for opp_stack in opp_stacks:
            state_vector.append(opp_stack / starting_stack)
        
        # Active players (normalized)
        state_vector.append(active_opponents / 5.0)
        
        # EV features
        breakeven_equity = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(breakeven_equity)
        excess_equity = equity - breakeven_equity
        state_vector.append(excess_equity)
        state_vector.append((starting_stack - my_stack) / starting_stack)
        state_vector.append(1.0 if to_call > 0 else 0.0)
        
        # Hand strength category (one-hot, 5 categories)
        strength_cat = compute_hand_strength_category(equity)
        for i in range(5):
            state_vector.append(1.0 if i == strength_cat else 0.0)
        
        # Street encoding (one-hot, 4 streets)
        street = [0.0, 0.0, 0.0, 0.0]
        if len(board) == 0: street[0] = 1.0
        elif len(board) == 3: street[1] = 1.0
        elif len(board) == 4: street[2] = 1.0
        else: street[3] = 1.0
        state_vector.extend(street)
        
        # V15: Position encoding (one-hot, 6 positions)
        position_encoding = [0.0] * 6
        position_encoding[agent_position] = 1.0
        state_vector.extend(position_encoding)
        
        # V15: Players to act after (normalized)
        # In position = advantage
        players_after = sum(1 for i in range(agent_position + 1, num_players) 
                           if players_in_hand[i])
        state_vector.append(players_after / 5.0)
        
        # V15: Players to act before
        players_before = sum(1 for i in range(0, agent_position) 
                            if players_in_hand[i])
        state_vector.append(players_before / 5.0)
        
        # Pad to fixed size (520)
        while len(state_vector) < 520:
            state_vector.append(0.0)
        
        return np.array(state_vector[:520], dtype=np.float32), {
            'equity': equity,
            'pot_bb': total_pot / big_blind,
            'to_call_bb': to_call / big_blind,
            'breakeven_equity': breakeven_equity,
            'excess_equity': excess_equity,
            'hand_strength': strength_cat,
            'my_stack': my_stack,
            'pot_size': total_pot,
            'position': agent_position,
            'active_opponents': active_opponents,
        }
    
    def get_legal_actions():
        """Get list of legal action indices."""
        legal = []
        if state.can_fold(): 
            legal.append(ACTION_FOLD)
        if state.can_check_or_call(): 
            legal.append(ACTION_CALL)
        if state.can_complete_bet_or_raise_to():
            legal.append(ACTION_RAISE_SMALL)
            legal.append(ACTION_RAISE_MEDIUM)
            legal.append(ACTION_RAISE_LARGE)
            legal.append(ACTION_ALL_IN)
        return legal if legal else [ACTION_CALL]
    
    def execute_action(action: int, actor: int) -> float:
        """Execute action with proper bet sizing. Returns amount risked."""
        amount_risked = 0.0
        
        if action == ACTION_FOLD:
            if state.can_fold(): 
                state.fold()
                players_in_hand[actor] = False
            elif state.can_check_or_call(): 
                state.check_or_call()
        
        elif action == ACTION_CALL:
            if state.can_check_or_call():
                current_bet = max(state.bets)
                my_bet = state.bets[actor]
                amount_risked = current_bet - my_bet
                state.check_or_call()
            elif state.can_fold(): 
                state.fold()
                players_in_hand[actor] = False
        
        elif action in [ACTION_RAISE_SMALL, ACTION_RAISE_MEDIUM, 
                        ACTION_RAISE_LARGE, ACTION_ALL_IN]:
            if state.can_complete_bet_or_raise_to():
                min_r = state.min_completion_betting_or_raising_to_amount
                max_r = state.max_completion_betting_or_raising_to_amount
                pot = sum(state.bets)
                my_bet = state.bets[actor]
                
                if action == ACTION_RAISE_SMALL:
                    amount = min(min_r * 2, max_r)
                elif action == ACTION_RAISE_MEDIUM:
                    amount = min(max(min_r, pot * 0.5), max_r)
                elif action == ACTION_RAISE_LARGE:
                    amount = min(max(min_r, pot), max_r)
                elif action == ACTION_ALL_IN:
                    amount = max_r
                
                amount_risked = amount - my_bet
                state.complete_bet_or_raise_to(int(amount))
            elif state.can_check_or_call(): 
                state.check_or_call()
        
        return amount_risked
    
    def opponent_action(opp_idx: int, player_idx: int, legal: List[int]) -> int:
        """Get opponent action."""
        opp = opponents[opp_idx]
        opp_hole = flatten_cards_list(state.hole_cards[player_idx])
        board = flatten_cards_list(state.board_cards)
        
        # Calculate equity vs remaining opponents
        num_active = count_active_opponents()
        equity = monte_carlo_equity_multiway(opp_hole, board, max(1, num_active - 1), 
                                              equity_iterations)
        
        total_pot = sum(state.bets)
        current_bet = max(state.bets)
        opp_bet = state.bets[player_idx]
        to_call = current_bet - opp_bet
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        
        return opp.select_action(
            equity=equity,
            pot_odds=pot_odds,
            to_call_bb=to_call / big_blind,
            pot_bb=total_pot / big_blind,
            legal_actions=legal,
            state=state,
            num_opponents=num_active
        )
    
    # Play episode
    transitions = []
    contexts = []
    actions_taken = []
    equities_taken = []
    amounts_risked = []
    pot_sizes_at_action = []
    action_counts = {i: 0 for i in range(NUM_ACTIONS_V15)}
    
    pending_obs = None
    pending_action = None
    pending_context = None
    
    # Map player index to opponent index (agent is not an opponent)
    opp_map = {}
    opp_idx = 0
    for i in range(num_players):
        if i != agent_position:
            opp_map[i] = opp_idx
            opp_idx += 1
    
    while state.status is not False:
        current_actor = state.actor_index
        
        # Skip if no actor (can happen during automations)
        if current_actor is None:
            run_automations()
            continue
        
        if current_actor == agent_position:  # Agent's turn
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
                    masked_q = torch.full_like(q_values, float('-inf'))
                    for a in legal:
                        masked_q[a] = q_values[a]
                    action = masked_q.argmax().item()
            
            action_counts[action] += 1
            pending_obs = obs
            pending_action = action
            pending_context = ctx
            
            equities_taken.append(ctx['equity'])
            pot_sizes_at_action.append(ctx['pot_size'])
            
            amount_risked = execute_action(action, actor=agent_position)
            amounts_risked.append(amount_risked)
            
            run_automations()
        
        else:  # Opponent's turn
            legal = get_legal_actions()
            opp_idx = opp_map.get(current_actor, 0)
            action = opponent_action(opp_idx, current_actor, legal)
            execute_action(action, actor=current_actor)
            run_automations()
    
    # Terminal
    final_reward = (state.stacks[agent_position] - starting_stack) / big_blind
    won_hand = final_reward > 0
    pot_won = state.stacks[agent_position] - starting_stack if won_hand else 0
    pot_won_bb = pot_won / big_blind
    
    # Calculate shaped reward
    if len(actions_taken) > 0 and len(equities_taken) > 0:
        last_action = actions_taken[-1] if actions_taken else ACTION_CALL
        last_equity = equities_taken[-1] if equities_taken else 0.5
        total_risked = sum(amounts_risked)
        risk_ratio = total_risked / starting_stack
        last_pot_size = pot_sizes_at_action[-1] / big_blind if pot_sizes_at_action else 0
        last_active_opponents = contexts[-1]['active_opponents'] if contexts else 1
        
        shaped_reward = compute_v15_shaped_reward(
            base_reward=final_reward,
            action=last_action,
            equity=last_equity,
            risk_ratio=risk_ratio,
            pot_won=pot_won_bb,
            pot_size_before_action=last_pot_size,
            won_hand=won_hand,
            actions_history=actions_taken,
            equities_history=equities_taken,
            position=agent_position,
            num_opponents_in_pot=last_active_opponents
        )
    else:
        shaped_reward = final_reward
    
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
        'shaped_reward': shaped_reward,
        'action_counts': action_counts,
        'position': agent_position,
        'position_name': POSITION_NAMES[agent_position],
        'won': won_hand,
        'equities': equities_taken,
        'amounts_risked': amounts_risked,
    }
