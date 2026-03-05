"""
Poker Worker V19: Pluribus-like Training

Key improvements over V17:
1. Three session modes: SCRIPTED, SELF_PLAY, MIXED
2. Phased training: 60% scripted early, then 33/33/34 split
3. Massive bust penalty (-200 BB)
4. 30 hands per session
"""

import random
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

from pokerkit import Automation, NoLimitTexasHoldem, Card, StandardHighHand, Deck
from poker_model_v19 import (
    DuelingPokerNet, NUM_ACTIONS_V19, OpponentPool,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE_33POT, 
    ACTION_RAISE_50POT, ACTION_RAISE_66POT, ACTION_RAISE_75POT, ACTION_RAISE_100POT, ACTION_RAISE_150POT, ACTION_ALL_IN,
    POS_UTG, POS_MP, POS_CO, POS_BTN, POS_SB, POS_BB,
    compute_hand_strength_category, compute_v19_hand_reward
    
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
    """Monte Carlo equity calculation against multiple opponents."""
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
        
        idx = 0
        opponent_hands = []
        for _ in range(num_opponents):
            opp_hole = deck_cards[idx:idx + cards_per_opponent]
            opponent_hands.append(opp_hole)
            idx += cards_per_opponent
        
        sim_board = board_cards + deck_cards[idx:idx + needed_board]
        
        my_total = hole_cards + sim_board
        my_hand = max(StandardHighHand(c) for c in combinations(my_total, 5))
        
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
                wins += 1.0 / (ties + 1)
            else:
                wins += 1
    
    return wins / iterations


# =============================================================================
# Scripted Opponent Classes (same as V15/V16/V17)
# =============================================================================

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
    def __init__(self):
        super().__init__('TAG')
        self.value_raise_threshold = 0.70
        self.call_threshold = 0.55
        self.bluff_frequency = 0.08
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        adjustment = num_opponents * 0.02
        
        if equity > self.value_raise_threshold + noise + adjustment:
            if ACTION_RAISE_100POT in legal_actions and equity > 0.80:
                return ACTION_RAISE_100POT
            if ACTION_RAISE_75POT in legal_actions:
                return ACTION_RAISE_75POT
            if ACTION_RAISE_50POT in legal_actions:
                return ACTION_RAISE_50POT
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if random.random() < self.bluff_frequency / (num_opponents + 1) and equity < 0.35:
            if ACTION_RAISE_75POT in legal_actions and pot_bb > 3:
                return ACTION_RAISE_75POT
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])


class LooseAggressiveOpponent(StochasticOpponent):
    def __init__(self):
        super().__init__('LAG')
        self.value_raise_threshold = 0.55
        self.call_threshold = 0.40
        self.bluff_frequency = 0.25
        self.overbet_frequency = 0.15
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        adjustment = num_opponents * 0.01
        
        if equity > self.value_raise_threshold + noise + adjustment:
            if equity > 0.75 and random.random() < self.overbet_frequency:
                if ACTION_ALL_IN in legal_actions:
                    return ACTION_ALL_IN
                if ACTION_RAISE_100POT in legal_actions:
                    return ACTION_RAISE_100POT
            
            if ACTION_RAISE_75POT in legal_actions:
                return ACTION_RAISE_75POT
            if ACTION_RAISE_50POT in legal_actions:
                return ACTION_RAISE_50POT
        
        if equity < 0.35 and random.random() < self.bluff_frequency / num_opponents:
            bluff_size = random.random()
            if bluff_size > 0.8 and ACTION_RAISE_100POT in legal_actions:
                return ACTION_RAISE_100POT
            if bluff_size > 0.4 and ACTION_RAISE_75POT in legal_actions:
                return ACTION_RAISE_75POT
            if ACTION_RAISE_50POT in legal_actions:
                return ACTION_RAISE_50POT
        
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])


class TightPassiveOpponent(StochasticOpponent):
    def __init__(self):
        super().__init__('Rock')
        self.monster_threshold = 0.85
        self.call_threshold = 0.60
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        adjustment = num_opponents * 0.03
        
        if equity > self.monster_threshold + noise:
            if ACTION_RAISE_75POT in legal_actions:
                return ACTION_RAISE_75POT
            if ACTION_RAISE_50POT in legal_actions:
                return ACTION_RAISE_50POT
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])


class LoosePassiveOpponent(StochasticOpponent):
    def __init__(self):
        super().__init__('CallingStation')
        self.fold_equity_threshold = 0.20
        self.raise_threshold = 0.80
        self.calling_tendency = 0.85
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        
        if equity > self.raise_threshold + noise:
            if ACTION_RAISE_50POT in legal_actions:
                return ACTION_RAISE_50POT
        
        if equity > self.fold_equity_threshold + noise:
            if ACTION_CALL in legal_actions and (to_call_bb == 0 or random.random() < self.calling_tendency):
                return ACTION_CALL
            if ACTION_CHECK in legal_actions:
                return ACTION_CHECK
        
        if equity < self.fold_equity_threshold and random.random() > 0.3:
            if ACTION_FOLD in legal_actions:
                return ACTION_FOLD
        
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])


class MixedBalancedOpponent(StochasticOpponent):
    def __init__(self):
        super().__init__('MixedBalanced')
        self.value_bet_ratio = 0.70
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        should_bet = random.random()
        adjustment = num_opponents * 0.02
        
        if equity > 0.75 + noise:
            if should_bet < 0.85:
                size_roll = random.random()
                if size_roll > 0.7 and ACTION_RAISE_100POT in legal_actions:
                    return ACTION_RAISE_100POT
                if size_roll > 0.3 and ACTION_RAISE_75POT in legal_actions:
                    return ACTION_RAISE_75POT
                if ACTION_RAISE_50POT in legal_actions:
                    return ACTION_RAISE_50POT
            return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])
        
        elif equity > 0.55 + noise + adjustment:
            if should_bet < 0.4:
                if ACTION_RAISE_50POT in legal_actions:
                    return ACTION_RAISE_50POT
            if equity > pot_odds + 0.05:
                return ACTION_CALL if ACTION_CALL in legal_actions else (ACTION_CHECK if ACTION_CHECK in legal_actions else legal_actions[0])
        
        elif equity > 0.40 + noise + adjustment:
            if equity > pot_odds:
                return ACTION_CALL if ACTION_CALL in legal_actions else (ACTION_CHECK if ACTION_CHECK in legal_actions else legal_actions[0])
        
        elif equity < 0.30:
            bluff_roll = random.random()
            if bluff_roll < 0.20 / num_opponents:
                if ACTION_RAISE_75POT in legal_actions:
                    return ACTION_RAISE_75POT
        
        if ACTION_FOLD in legal_actions:
            return ACTION_FOLD
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])


def create_opponent(opp_type: str) -> StochasticOpponent:
    opponents = {
        'TAG': TightAggressiveOpponent,
        'LAG': LooseAggressiveOpponent,
        'Rock': TightPassiveOpponent,
        'CallingStation': LoosePassiveOpponent,
        'MixedBalanced': MixedBalancedOpponent,
    }
    return opponents.get(opp_type, MixedBalancedOpponent)()


POSITION_NAMES = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']


# =============================================================================
# Self-Play Opponent Wrapper
# =============================================================================

class SelfPlayOpponent:
    def __init__(self, model: Optional[DuelingPokerNet], epsilon: float = 0.05):
        self.model = model
        self.opp_type = 'SelfPlay'
        self.epsilon = epsilon
    
    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        if self.model is None or random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.model(state_tensor).squeeze(0)
            masked_q = torch.full_like(q_values, float('-inf'))
            for a in legal_actions:
                masked_q[a] = q_values[a]
            return masked_q.argmax().item()


# =============================================================================
# V18 Hybrid Session Runner
# =============================================================================


def run_training_session_v19(args):
    """
    100% Self-Play training session.
    """
    (session_seed, epsilon, equity_iterations, model_state_dict, 
     hands_per_session, opponent_pool_state_dicts, session_num) = args
    
    random.seed(session_seed)
    
    model = DuelingPokerNet(state_dim=520)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    num_players = 6
    starting_stack_bb = 100.0
    big_blind = 10
    small_blind = 5
    
    agent_stack = starting_stack_bb * big_blind
    session_transitions = []
    session_action_counts = {i: 0 for i in range(NUM_ACTIONS_V19)}
    allin_outcomes = []
    hands_completed = 0
    position_history = []
    hand_profits = []
    
    # 100% self-play: get 5 opponents from the pool
    opponents = []
    for _ in range(5):
        if not opponent_pool_state_dicts:
            # Random opponents at start
            opponents.append(SelfPlayOpponent(None, epsilon=1.0))
        else:
            opp_state = random.choice(opponent_pool_state_dicts)
            opp_model = DuelingPokerNet(state_dim=520)
            opp_model.load_state_dict(opp_state)
            opp_model.eval()
            opponents.append(SelfPlayOpponent(opp_model, epsilon=0.05))
            
    for hand_num in range(hands_per_session):
        agent_stack_bb = agent_stack / big_blind
        if agent_stack_bb < 5.0:
            break
            
        # Random position
        agent_position = random.randint(0, 5)
        position_history.append(agent_position)
        
        random.shuffle(opponents)
        
        stacks = [1000] * num_players
        stacks[agent_position] = int(agent_stack)
        
        try:
            state = NoLimitTexasHoldem.create_state(
                automations=(Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                             Automation.BLIND_OR_STRADDLE_POSTING, 
                             Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                             Automation.HAND_KILLING, Automation.CHIPS_PUSHING, 
                             Automation.CHIPS_PULLING,),
                ante_trimming_status=True, raw_antes={-1: 0},
                raw_blinds_or_straddles=(small_blind, big_blind),
                min_bet=big_blind,
                raw_starting_stacks=stacks,
                player_count=num_players,
            )
        except Exception:
            break
            
        while state.can_deal_hole(): state.deal_hole()
        
        def run_automations():
            while state.can_burn_card(): state.burn_card('??')
            while state.can_deal_board(): state.deal_board()
            while state.can_push_chips(): state.push_chips()
            while state.can_pull_chips(): state.pull_chips()
            
        run_automations()
        players_in_hand = [True] * num_players
        
        def get_observation(player_idx: int = None, compute_equity: bool = True) -> Tuple[np.ndarray, Dict]:
            if player_idx is None: player_idx = agent_position
            
            state_vector = []
            hole = flatten_cards_list(state.hole_cards[player_idx])
            board = flatten_cards_list(state.board_cards)
            
            ranks, suits = '23456789TJQKA', 'cdhs'
            
            for i in range(2):
                encoding = np.zeros(52, dtype=np.float32)
                if i < len(hole):
                    idx = ranks.index(hole[i].rank) * 4 + suits.index(hole[i].suit)
                    encoding[idx] = 1.0
                state_vector.extend(encoding)
            
            for i in range(5):
                encoding = np.zeros(52, dtype=np.float32)
                if i < len(board):
                    idx = ranks.index(board[i].rank) * 4 + suits.index(board[i].suit)
                    encoding[idx] = 1.0
                state_vector.extend(encoding)
            
            total_pot = sum(state.bets)
            current_bet = max(state.bets)
            my_bet = state.bets[player_idx]
            to_call = current_bet - my_bet
            my_stack = state.stacks[player_idx]
            
            opp_stacks = [state.stacks[i] for i in range(num_players) if i != player_idx]
            active_opponents = sum(1 for i, active in enumerate(players_in_hand) if active and i != player_idx)
            
            if compute_equity:
                equity = monte_carlo_equity_multiway(hole, board, max(1, active_opponents), equity_iterations)
            else:
                equity = 0.5
                
            state_vector.append(equity)
            
            pot_odds = to_call / (total_pot + to_call + 1e-6)
            state_vector.append(pot_odds)
            state_vector.append(min(to_call / (total_pot + 1e-6), 2.0))
            state_vector.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
            state_vector.append(my_stack / (starting_stack_bb * big_blind))
            state_vector.append(total_pot / (starting_stack_bb * big_blind * num_players))
            
            # Fix: opponent stacks normalized properly
            for opp_stack in opp_stacks:
                state_vector.append(opp_stack / (starting_stack_bb * big_blind))
            
            state_vector.append(active_opponents / 5.0)
            
            breakeven_equity = to_call / (total_pot + to_call + 1e-6)
            state_vector.append(breakeven_equity)
            state_vector.append(equity - breakeven_equity)
            state_vector.append((starting_stack_bb * big_blind - my_stack) / (starting_stack_bb * big_blind))
            state_vector.append(1.0 if to_call > 0 else 0.0)
            
            strength_cat = compute_hand_strength_category(equity)
            for i in range(5):
                state_vector.append(1.0 if i == strength_cat else 0.0)
            
            street = [0.0] * 4
            if len(board) == 0: street[0] = 1.0
            elif len(board) == 3: street[1] = 1.0
            elif len(board) == 4: street[2] = 1.0
            else: street[3] = 1.0
            state_vector.extend(street)
            
            # Fix: always encode agent's own position
            position_encoding = [0.0] * 6
            position_encoding[agent_position] = 1.0
            state_vector.extend(position_encoding)
            
            players_after = sum(1 for i in range(player_idx + 1, num_players) if players_in_hand[i])
            state_vector.append(players_after / 5.0)
            
            players_before = sum(1 for i in range(0, player_idx) if players_in_hand[i])
            state_vector.append(players_before / 5.0)
            
            while len(state_vector) < 520: state_vector.append(0.0)
            
            return np.array(state_vector[:520], dtype=np.float32), {
                'equity': equity,
                'pot_bb': total_pot / big_blind,
                'position': player_idx,
                'active_opponents': active_opponents,
            }

        def get_legal_actions():
            legal = []
            
            # Safely check actor index
            try:
                actor = state.actor_index
                if actor is None: actor = agent_position # Fallback
                to_call = max(state.bets) - state.bets[actor]
            except Exception:
                to_call = 0
                
            if to_call > 0:
                if state.can_fold(): legal.append(ACTION_FOLD)
                if state.can_check_or_call(): legal.append(ACTION_CALL)
            else:
                if state.can_check_or_call(): legal.append(ACTION_CHECK)
                
            if state.can_complete_bet_or_raise_to():
                legal.extend([ACTION_RAISE_33POT, ACTION_RAISE_50POT, ACTION_RAISE_66POT, ACTION_RAISE_75POT, ACTION_RAISE_100POT, ACTION_RAISE_150POT, ACTION_ALL_IN])
            
            # Fallback just in case
            if not legal:
                legal = [ACTION_CHECK] if to_call == 0 else [ACTION_CALL]
                
            return legal

        def execute_action(action: int, actor: int):
            if action == ACTION_FOLD:
                if state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
                elif state.can_check_or_call(): state.check_or_call()
            elif action in [ACTION_CALL, ACTION_CHECK]:
                if state.can_check_or_call(): state.check_or_call()
                elif state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
            elif action in [ACTION_RAISE_33POT, ACTION_RAISE_50POT, ACTION_RAISE_66POT, ACTION_RAISE_75POT, ACTION_RAISE_100POT, ACTION_RAISE_150POT, ACTION_ALL_IN]:
                if state.can_complete_bet_or_raise_to():
                    min_r = state.min_completion_betting_or_raising_to_amount
                    max_r = state.max_completion_betting_or_raising_to_amount
                    pot = sum(state.bets)
                    if action == ACTION_RAISE_33POT: amount = min(max(min_r, int(pot * 0.33)), max_r)
                    elif action == ACTION_RAISE_50POT: amount = min(max(min_r, int(pot * 0.50)), max_r)
                    elif action == ACTION_RAISE_75POT: amount = min(max(min_r, int(pot * 0.75)), max_r)
                    elif action == ACTION_RAISE_100POT: amount = min(max(min_r, int(pot * 1.0)), max_r)
                    elif action == ACTION_RAISE_150POT: amount = min(max(min_r, int(pot * 1.50)), max_r)
                    elif action == ACTION_RAISE_66POT: amount = min(max(min_r, int(pot * 0.66)), max_r)
                    elif action == ACTION_ALL_IN: amount = max_r
                    state.complete_bet_or_raise_to(int(amount))
                elif state.can_check_or_call(): state.check_or_call()

        opp_map = {i: opp_idx for opp_idx, i in enumerate(i for i in range(num_players) if i != agent_position)}
        
        hand_transitions = []
        hand_actions = []
        pending_obs = None
        pending_action = None
        stack_before_hand = agent_stack
        
        while state.status is not False:
            current_actor = state.actor_index
            if current_actor is None:
                run_automations()
                continue
                
            if current_actor == agent_position:
                obs, ctx = get_observation(compute_equity=True)
                legal = get_legal_actions()
                
                if pending_obs is not None:
                    # Non-terminal transitions get reward 0.0
                    hand_transitions.append((pending_obs, pending_action, 0.0, obs, False))
                
                if random.random() < epsilon:
                    action = random.choice(legal)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        q_values = model(state_tensor).squeeze(0)
                        masked_q = torch.full_like(q_values, float('-inf'))
                        for a in legal: masked_q[a] = q_values[a]
                        action = masked_q.argmax().item()
                        
                session_action_counts[action] += 1
                pending_obs = obs
                pending_action = action
                hand_actions.append(action)
                execute_action(action, actor=agent_position)
                run_automations()
            else:
                legal = get_legal_actions()
                opp_idx = opp_map.get(current_actor, 0)
                opp = opponents[opp_idx]
                # self-play opponents: NO equity calculation needed
                obs, _ = get_observation(current_actor, compute_equity=False)
                action = opp.select_action(obs, legal)
                execute_action(action, actor=current_actor)
                run_automations()
                
        hand_profit = state.stacks[agent_position] - stack_before_hand
        hand_profit_bb = hand_profit / big_blind
        agent_stack = state.stacks[agent_position]
        hand_profits.append(hand_profit_bb)
        hands_completed += 1
        
        if pending_obs is not None:
            term_obs, _ = get_observation(compute_equity=False)
            reward_bb = compute_v19_hand_reward(hand_profit_bb) # pure profit/loss
            hand_transitions.append((pending_obs, pending_action, reward_bb, term_obs, True))
            
        session_transitions.extend(hand_transitions)
        
    final_stack_bb = agent_stack / big_blind
    session_profit_bb = final_stack_bb - starting_stack_bb
    
    return {
        'transitions': session_transitions,
        'session_reward': session_profit_bb, # raw session profit
        'session_profit_bb': session_profit_bb,
        'final_stack_bb': final_stack_bb,
        'hands_completed': hands_completed,
        'session_completed': True,
        'busted': final_stack_bb < 5.0,
        'action_counts': session_action_counts,
        'allin_outcomes': allin_outcomes,
        'position_history': position_history,
        'hand_profits': hand_profits,
        'opponent_mode': 'self-play',
        'session_mode': 'self_play',
    }


def run_eval_session_v19(args):
    """
    Eval session against scripted opponents. No transitions returned.
    """
    (session_seed, epsilon, equity_iterations, model_state_dict, hands_per_session) = args
    
    random.seed(session_seed)
    
    model = DuelingPokerNet(state_dim=520)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    num_players = 6
    starting_stack_bb = 100.0
    big_blind = 10
    small_blind = 5
    
    agent_stack = starting_stack_bb * big_blind
    session_action_counts = {i: 0 for i in range(NUM_ACTIONS_V19)}
    hands_completed = 0
    hand_profits = []
    
    from poker_worker_v18 import create_opponent # ensure it exists
    scripted_types = ['TAG', 'LAG', 'Rock', 'CallingStation', 'MixedBalanced']
    
    for hand_num in range(hands_per_session):
        agent_stack_bb = agent_stack / big_blind
        if agent_stack_bb < 5.0:
            break
            
        agent_position = hand_num % 6
        
        opponents = [create_opponent(t) for t in scripted_types]
        random.shuffle(opponents)
        
        stacks = [1000] * num_players
        stacks[agent_position] = int(agent_stack)
        
        try:
            state = NoLimitTexasHoldem.create_state(
                automations=(Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                             Automation.BLIND_OR_STRADDLE_POSTING, 
                             Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                             Automation.HAND_KILLING, Automation.CHIPS_PUSHING, 
                             Automation.CHIPS_PULLING,),
                ante_trimming_status=True, raw_antes={-1: 0},
                raw_blinds_or_straddles=(small_blind, big_blind),
                min_bet=big_blind,
                raw_starting_stacks=stacks,
                player_count=num_players,
            )
        except Exception:
            break
            
        while state.can_deal_hole(): state.deal_hole()
        
        def run_automations():
            while state.can_burn_card(): state.burn_card('??')
            while state.can_deal_board(): state.deal_board()
            while state.can_push_chips(): state.push_chips()
            while state.can_pull_chips(): state.pull_chips()
            
        run_automations()
        players_in_hand = [True] * num_players
        
        def count_active_opponents():
            return sum(1 for i, active in enumerate(players_in_hand) if active and i != agent_position)

        def get_observation(player_idx: int = None) -> Tuple[np.ndarray, Dict]:
            if player_idx is None: player_idx = agent_position
            state_vector = []
            hole = flatten_cards_list(state.hole_cards[player_idx])
            board = flatten_cards_list(state.board_cards)
            ranks, suits = '23456789TJQKA', 'cdhs'
            for i in range(2):
                encoding = np.zeros(52, dtype=np.float32)
                if i < len(hole):
                    idx = ranks.index(hole[i].rank) * 4 + suits.index(hole[i].suit)
                    encoding[idx] = 1.0
                state_vector.extend(encoding)
            for i in range(5):
                encoding = np.zeros(52, dtype=np.float32)
                if i < len(board):
                    idx = ranks.index(board[i].rank) * 4 + suits.index(board[i].suit)
                    encoding[idx] = 1.0
                state_vector.extend(encoding)
            total_pot = sum(state.bets)
            current_bet = max(state.bets)
            to_call = current_bet - state.bets[player_idx]
            my_stack = state.stacks[player_idx]
            opp_stacks = [state.stacks[i] for i in range(num_players) if i != player_idx]
            active_opponents = max(1, sum(1 for i, active in enumerate(players_in_hand) if active and i != player_idx))
            equity = monte_carlo_equity_multiway(hole, board, active_opponents, equity_iterations)
            state_vector.append(equity)
            pot_odds = to_call / (total_pot + to_call + 1e-6)
            state_vector.append(pot_odds)
            state_vector.append(min(to_call / (total_pot + 1e-6), 2.0))
            state_vector.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
            state_vector.append(my_stack / (starting_stack_bb * big_blind))
            state_vector.append(total_pot / (starting_stack_bb * big_blind * num_players))
            for opp_stack in opp_stacks:
                state_vector.append(opp_stack / (starting_stack_bb * big_blind))
            state_vector.append(active_opponents / 5.0)
            breakeven_equity = to_call / (total_pot + to_call + 1e-6)
            state_vector.append(breakeven_equity)
            state_vector.append(equity - breakeven_equity)
            state_vector.append((starting_stack_bb * big_blind - my_stack) / (starting_stack_bb * big_blind))
            state_vector.append(1.0 if to_call > 0 else 0.0)
            strength_cat = compute_hand_strength_category(equity)
            for i in range(5): state_vector.append(1.0 if i == strength_cat else 0.0)
            street = [0.0] * 4
            if len(board) == 0: street[0] = 1.0
            elif len(board) == 3: street[1] = 1.0
            elif len(board) == 4: street[2] = 1.0
            else: street[3] = 1.0
            state_vector.extend(street)
            position_encoding = [0.0] * 6
            position_encoding[agent_position] = 1.0
            state_vector.extend(position_encoding)
            players_after = sum(1 for i in range(player_idx + 1, num_players) if players_in_hand[i])
            state_vector.append(players_after / 5.0)
            players_before = sum(1 for i in range(0, player_idx) if players_in_hand[i])
            state_vector.append(players_before / 5.0)
            while len(state_vector) < 520: state_vector.append(0.0)
            return np.array(state_vector[:520], dtype=np.float32), {'equity': equity}

        def get_legal_actions():
            legal = []
            
            # Safely check actor index
            try:
                actor = state.actor_index
                if actor is None: actor = agent_position # Fallback
                to_call = max(state.bets) - state.bets[actor]
            except Exception:
                to_call = 0
                
            if to_call > 0:
                if state.can_fold(): legal.append(ACTION_FOLD)
                if state.can_check_or_call(): legal.append(ACTION_CALL)
            else:
                if state.can_check_or_call(): legal.append(ACTION_CHECK)
                
            if state.can_complete_bet_or_raise_to():
                legal.extend([ACTION_RAISE_33POT, ACTION_RAISE_50POT, ACTION_RAISE_66POT, ACTION_RAISE_75POT, ACTION_RAISE_100POT, ACTION_RAISE_150POT, ACTION_ALL_IN])
            
            # Fallback just in case
            if not legal:
                legal = [ACTION_CHECK] if to_call == 0 else [ACTION_CALL]
                
            return legal

        def execute_action(action: int, actor: int):
            if action == ACTION_FOLD:
                if state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
                elif state.can_check_or_call(): state.check_or_call()
            elif action in [ACTION_CALL, ACTION_CHECK]:
                if state.can_check_or_call(): state.check_or_call()
                elif state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
            elif action in [ACTION_RAISE_33POT, ACTION_RAISE_50POT, ACTION_RAISE_66POT, ACTION_RAISE_75POT, ACTION_RAISE_100POT, ACTION_RAISE_150POT, ACTION_ALL_IN]:
                if state.can_complete_bet_or_raise_to():
                    min_r = state.min_completion_betting_or_raising_to_amount
                    max_r = state.max_completion_betting_or_raising_to_amount
                    pot = sum(state.bets)
                    if action == ACTION_RAISE_33POT: amount = min(max(min_r, int(pot * 0.33)), max_r)
                    elif action == ACTION_RAISE_50POT: amount = min(max(min_r, int(pot * 0.50)), max_r)
                    elif action == ACTION_RAISE_75POT: amount = min(max(min_r, int(pot * 0.75)), max_r)
                    elif action == ACTION_RAISE_100POT: amount = min(max(min_r, int(pot * 1.0)), max_r)
                    elif action == ACTION_RAISE_150POT: amount = min(max(min_r, int(pot * 1.50)), max_r)
                    elif action == ACTION_RAISE_66POT: amount = min(max(min_r, int(pot * 0.66)), max_r)
                    elif action == ACTION_ALL_IN: amount = max_r
                    state.complete_bet_or_raise_to(int(amount))
                elif state.can_check_or_call(): state.check_or_call()

        def scripted_opponent_action(opp, player_idx, legal):
            opp_hole = flatten_cards_list(state.hole_cards[player_idx])
            board = flatten_cards_list(state.board_cards)
            num_active = count_active_opponents()
            equity = monte_carlo_equity_multiway(opp_hole, board, max(1, num_active - 1), equity_iterations)
            total_pot = sum(state.bets)
            to_call = max(state.bets) - state.bets[player_idx]
            pot_odds = to_call / (total_pot + to_call + 1e-6)
            return opp.select_action(equity=equity, pot_odds=pot_odds, to_call_bb=to_call / big_blind, pot_bb=total_pot / big_blind, legal_actions=legal, state=state, num_opponents=num_active)
            
        opp_map = {i: opp_idx for opp_idx, i in enumerate(i for i in range(num_players) if i != agent_position)}
        stack_before_hand = agent_stack
        
        while state.status is not False:
            current_actor = state.actor_index
            if current_actor is None:
                run_automations()
                continue
                
            if current_actor == agent_position:
                obs, _ = get_observation()
                legal = get_legal_actions()
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = model(state_tensor).squeeze(0)
                    masked_q = torch.full_like(q_values, float('-inf'))
                    for a in legal: masked_q[a] = q_values[a]
                    action = masked_q.argmax().item()
                        
                session_action_counts[action] += 1
                execute_action(action, actor=agent_position)
                run_automations()
            else:
                legal = get_legal_actions()
                opp_idx = opp_map.get(current_actor, 0)
                opp = opponents[opp_idx]
                action = scripted_opponent_action(opp, current_actor, legal)
                execute_action(action, actor=current_actor)
                run_automations()
                
        hand_profit = state.stacks[agent_position] - stack_before_hand
        hand_profit_bb = hand_profit / big_blind
        agent_stack = state.stacks[agent_position]
        hand_profits.append(hand_profit_bb)
        hands_completed += 1
        
    final_stack_bb = agent_stack / big_blind
    session_profit_bb = final_stack_bb - starting_stack_bb
    
    return {
        'session_profit_bb': session_profit_bb,
        'final_stack_bb': final_stack_bb,
        'hands_completed': hands_completed,
        'busted': final_stack_bb < 5.0,
        'action_counts': session_action_counts,
        'hand_profits': hand_profits,
    }
