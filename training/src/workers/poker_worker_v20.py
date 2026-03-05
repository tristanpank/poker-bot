"""
Poker Worker V20: Pluribus-like Training

Key improvements over V18:
1. Removed session logic.
2. Only independent randomized position hands.
3. 7 fixed action sizes (Fold, Check, Call, 33%, 66%, 100%, All-in).
"""

import random
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

from pokerkit import Automation, NoLimitTexasHoldem, Card, StandardHighHand, Deck
from poker_model_v20 import (
    DuelingPokerNet, NUM_ACTIONS_V20, OpponentPool,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL, ACTION_RAISE_33POT, 
    ACTION_RAISE_66POT, ACTION_RAISE_100POT, ACTION_ALL_IN,
    POS_UTG, POS_MP, POS_CO, POS_BTN, POS_SB, POS_BB,
    compute_hand_strength_category, compute_v20_hand_reward
)

def flatten_cards_list(items):
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
# Scripted Opponent Classes (simplified for 7 actions)
# =============================================================================

class StochasticOpponent:
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
            if ACTION_RAISE_66POT in legal_actions:
                return ACTION_RAISE_66POT
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if random.random() < self.bluff_frequency / (num_opponents + 1) and equity < 0.35:
            if ACTION_RAISE_66POT in legal_actions and pot_bb > 3:
                return ACTION_RAISE_66POT
        
        if ACTION_FOLD in legal_actions: return ACTION_FOLD
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
                if ACTION_ALL_IN in legal_actions: return ACTION_ALL_IN
                if ACTION_RAISE_100POT in legal_actions: return ACTION_RAISE_100POT
            if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
        
        if equity < 0.35 and random.random() < self.bluff_frequency / num_opponents:
            bluff_size = random.random()
            if bluff_size > 0.8 and ACTION_RAISE_100POT in legal_actions: return ACTION_RAISE_100POT
            if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
        
        if equity > self.call_threshold + noise:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if ACTION_FOLD in legal_actions: return ACTION_FOLD
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
            if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
        
        if equity > self.call_threshold + noise + adjustment:
            if ACTION_CALL in legal_actions: return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if ACTION_FOLD in legal_actions: return ACTION_FOLD
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
            if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
        
        if equity > self.fold_equity_threshold + noise:
            if ACTION_CALL in legal_actions and (to_call_bb == 0 or random.random() < self.calling_tendency): return ACTION_CALL
            if ACTION_CHECK in legal_actions: return ACTION_CHECK
        
        if equity < self.fold_equity_threshold and random.random() > 0.3:
            if ACTION_FOLD in legal_actions: return ACTION_FOLD
        
        return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])

class MixedBalancedOpponent(StochasticOpponent):
    def __init__(self):
        super().__init__('MixedBalanced')
    
    def select_action(self, equity, pot_odds, to_call_bb, pot_bb, legal_actions, state, num_opponents=1):
        noise = self.noise()
        should_bet = random.random()
        adjustment = num_opponents * 0.02
        
        if equity > 0.75 + noise:
            if should_bet < 0.85:
                size_roll = random.random()
                if size_roll > 0.7 and ACTION_RAISE_100POT in legal_actions: return ACTION_RAISE_100POT
                if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
            return ACTION_CHECK if ACTION_CHECK in legal_actions else (ACTION_CALL if ACTION_CALL in legal_actions else legal_actions[0])
        
        elif equity > 0.55 + noise + adjustment:
            if should_bet < 0.4:
                if ACTION_RAISE_33POT in legal_actions: return ACTION_RAISE_33POT
            if equity > pot_odds + 0.05:
                return ACTION_CALL if ACTION_CALL in legal_actions else (ACTION_CHECK if ACTION_CHECK in legal_actions else legal_actions[0])
        
        elif equity > 0.40 + noise + adjustment:
            if equity > pot_odds:
                return ACTION_CALL if ACTION_CALL in legal_actions else (ACTION_CHECK if ACTION_CHECK in legal_actions else legal_actions[0])
        
        elif equity < 0.30:
            if random.random() < 0.20 / num_opponents:
                if ACTION_RAISE_66POT in legal_actions: return ACTION_RAISE_66POT
        
        if ACTION_FOLD in legal_actions: return ACTION_FOLD
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
# Hands Batch Runner
# =============================================================================

def run_training_batch_v20(args):
    """
    Simulates a batch of independent poker hands. Stacks are reset every hand.
    """
    (batch_seed, epsilon, equity_iterations, model_state_dict, 
     hands_to_play, opponent_pool_state_dicts, batch_num) = args
    
    random.seed(batch_seed)
    
    import time
    perf_stats = {
        'total_time': 0.0,
        'sim_time': 0.0,
        'nn_time': 0.0,
        'mc_equity_time': 0.0,
        'overhead_time': 0.0
    }
    t_start_batch = time.time()
    
    model = DuelingPokerNet(state_dim=520)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    num_players = 6
    starting_stack_bb = 100.0
    big_blind = 10
    small_blind = 5
    
    batch_transitions = []
    action_counts = {i: 0 for i in range(NUM_ACTIONS_V20)}
    hand_profits = []
    position_history = []
    equity_cache = {}
    
    vpip_count = 0
    pfr_count = 0
    three_bet_count = 0
    
    # Pre-select opponents from pool
    opponents = []
    for _ in range(5):
        if not opponent_pool_state_dicts:
            opponents.append(SelfPlayOpponent(None, epsilon=1.0))
        else:
            opp_state = random.choice(opponent_pool_state_dicts)
            opp_model = DuelingPokerNet(state_dim=520)
            opp_model.load_state_dict(opp_state)
            opp_model.eval()
            opponents.append(SelfPlayOpponent(opp_model, epsilon=0.05))
            
    for hand_num in range(hands_to_play):
        agent_position = random.randint(0, 5)
        position_history.append(agent_position)
        random.shuffle(opponents)
        
        stacks = []
        for _ in range(num_players):
            bb_stack = max(50.0, min(200.0, random.gauss(starting_stack_bb, 20.0)))
            stacks.append(int(bb_stack * big_blind))
            
        agent_starting_stack = stacks[agent_position]
        
        t_sim_start = time.time()
        try:
            state = NoLimitTexasHoldem.create_state(
                automations=(Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                             Automation.BLIND_OR_STRADDLE_POSTING, 
                             Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                             Automation.HAND_KILLING, Automation.CHIPS_PUSHING, 
                             Automation.CHIPS_PULLING, Automation.CARD_BURNING,
                             Automation.BOARD_DEALING),
                ante_trimming_status=True, raw_antes={-1: 0},
                raw_blinds_or_straddles=(small_blind, big_blind),
                min_bet=big_blind,
                raw_starting_stacks=stacks,
                player_count=num_players,
            )
        except Exception:
            break
            
        while state.can_deal_hole(): state.deal_hole()
        players_in_hand = [True] * num_players
        perf_stats['sim_time'] += (time.time() - t_sim_start)
        
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
                active_num = max(1, active_opponents)
                hole_key = tuple(sorted(str(c) for c in hole))
                board_key = tuple(sorted(str(c) for c in board))
                cache_key = (hole_key, board_key, active_num)
                
                t_mc_start = time.time()
                if cache_key in equity_cache:
                    equity = equity_cache[cache_key]
                else:
                    equity = monte_carlo_equity_multiway(hole, board, active_num, equity_iterations)
                    equity_cache[cache_key] = equity
                perf_stats['mc_equity_time'] += (time.time() - t_mc_start)
            else:
                equity = 0.5
                
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
            state_vector.append(1.0 if to_call > 0 else 0.0)
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
            
            position_encoding = [0.0] * 6
            position_encoding[agent_position] = 1.0
            state_vector.extend(position_encoding)
            
            players_after = sum(1 for i in range(player_idx + 1, num_players) if players_in_hand[i])
            state_vector.append(players_after / 5.0)
            
            players_before = sum(1 for i in range(0, player_idx) if players_in_hand[i])
            state_vector.append(players_before / 5.0)
            
            if len(state_vector) < 520:
                state_vector.extend([0.0] * (520 - len(state_vector)))
            
            return np.array(state_vector[:520], dtype=np.float32), {
                'equity': equity,
                'pot_bb': total_pot / big_blind,
                'position': player_idx,
                'active_opponents': active_opponents,
            }

        def get_legal_actions():
            legal = []
            try:
                actor = state.actor_index
                if actor is None: actor = agent_position
                to_call = max(state.bets) - state.bets[actor]
            except Exception:
                to_call = 0
                
            if to_call > 0:
                # Facing a bet: can fold or call
                if state.can_fold(): legal.append(ACTION_FOLD)
                if state.can_check_or_call(): legal.append(ACTION_CALL)
            else:
                # No bet to call: can check
                if state.can_check_or_call(): legal.append(ACTION_CHECK)
                
            if state.can_complete_bet_or_raise_to():
                legal.extend([ACTION_RAISE_33POT, ACTION_RAISE_66POT, ACTION_RAISE_100POT, ACTION_ALL_IN])
            
            # Fallback safety
            if not legal:
                legal = [ACTION_CHECK] if to_call == 0 else [ACTION_FOLD]
                
            return legal

        # Stats tracking
        hand_vpip = False
        hand_pfr = False
        hand_3bet = False
        is_preflop = True

        def execute_action(action: int, actor: int):
            nonlocal hand_vpip, hand_pfr, hand_3bet, is_preflop
            
            is_preflop = len(state.board_cards) == 0
            
            # Track agent preflop stats
            if actor == agent_position and is_preflop:
                if action in [ACTION_CALL, ACTION_RAISE_33POT, ACTION_RAISE_66POT, ACTION_RAISE_100POT, ACTION_ALL_IN]:
                    hand_vpip = True
                
                if action in [ACTION_RAISE_33POT, ACTION_RAISE_66POT, ACTION_RAISE_100POT, ACTION_ALL_IN]:
                    hand_pfr = True
                    # A 3-bet happens if there's already been a raise preflop
                    # (big_blind + some original raise)
                    if max(state.bets) > big_blind:
                        hand_3bet = True

            if action == ACTION_FOLD:
                if state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
                elif state.can_check_or_call():
                    state.check_or_call()  # can't fold, just check
            elif action == ACTION_CHECK:
                if state.can_check_or_call():
                    state.check_or_call()
            elif action == ACTION_CALL:
                if state.can_check_or_call():
                    state.check_or_call()
                elif state.can_fold(): 
                    state.fold()
                    players_in_hand[actor] = False
            elif action in [ACTION_RAISE_33POT, ACTION_RAISE_66POT, ACTION_RAISE_100POT, ACTION_ALL_IN]:
                if state.can_complete_bet_or_raise_to():
                    min_r = state.min_completion_betting_or_raising_to_amount
                    max_r = state.max_completion_betting_or_raising_to_amount
                    pot = sum(state.bets)
                    if action == ACTION_RAISE_33POT: amount = min(max(min_r, int(pot * 0.33)), max_r)
                    elif action == ACTION_RAISE_66POT: amount = min(max(min_r, int(pot * 0.66)), max_r)
                    elif action == ACTION_RAISE_100POT: amount = min(max(min_r, int(pot * 1.0)), max_r)
                    elif action == ACTION_ALL_IN: amount = max_r
                    state.complete_bet_or_raise_to(int(amount))
                elif state.can_check_or_call(): state.check_or_call()

        opp_map = {i: opp_idx for opp_idx, i in enumerate(i for i in range(num_players) if i != agent_position)}
        
        hand_transitions = []
        hand_actions = []
        pending_obs = None
        pending_action = None
        
        # Reset per-hand stats
        hand_vpip = False
        hand_pfr = False
        hand_3bet = False
        
        while state.status is not False:
            t_sim_start = time.time()
            current_actor = state.actor_index
            if current_actor is None:
                perf_stats['sim_time'] += (time.time() - t_sim_start)
                continue
            perf_stats['sim_time'] += (time.time() - t_sim_start)
                
            if current_actor == agent_position:
                obs, ctx = get_observation(compute_equity=True)
                legal = get_legal_actions()
                
                if pending_obs is not None:
                    # Non-terminal transitions get reward 0.0
                    hand_transitions.append((pending_obs, pending_action, 0.0, obs, False))
                
                if random.random() < epsilon:
                    action = random.choice(legal)
                else:
                    t_nn_start = time.time()
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        q_values = model(state_tensor).squeeze(0)
                        masked_q = torch.full_like(q_values, float('-inf'))
                        for a in legal: masked_q[a] = q_values[a]
                        action = masked_q.argmax().item()
                    perf_stats['nn_time'] += (time.time() - t_nn_start)
                        
                action_counts[action] += 1
                pending_obs = obs
                pending_action = action
                hand_actions.append(action)
                
                t_sim_start = time.time()
                execute_action(action, actor=agent_position)
                perf_stats['sim_time'] += (time.time() - t_sim_start)
            else:
                legal = get_legal_actions()
                opp_idx = opp_map.get(current_actor, 0)
                opp = opponents[opp_idx]
                
                # In V20 fast_mode, opponents still need observations because they use NN models
                obs, _ = get_observation(current_actor, compute_equity=False)
                
                t_nn_start = time.time()
                action = opp.select_action(obs, legal)
                perf_stats['nn_time'] += (time.time() - t_nn_start)
                
                t_sim_start = time.time()
                execute_action(action, actor=current_actor)
                perf_stats['sim_time'] += (time.time() - t_sim_start)
                
        # Hand over, measure pure profit
        final_stack = state.stacks[agent_position]
        hand_profit = final_stack - agent_starting_stack
        hand_profit_bb = hand_profit / big_blind
        hand_profits.append(hand_profit_bb)
        
        if pending_obs is not None:
            term_obs, _ = get_observation(compute_equity=False)
            reward_bb = compute_v20_hand_reward(hand_profit_bb)
            hand_transitions.append((pending_obs, pending_action, reward_bb, term_obs, True))
            
        batch_transitions.extend(hand_transitions)
        
        vpip_count += 1 if hand_vpip else 0
        pfr_count += 1 if hand_pfr else 0
        three_bet_count += 1 if hand_3bet else 0
        
    perf_stats['total_time'] = time.time() - t_start_batch
    perf_stats['overhead_time'] = max(0.0, perf_stats['total_time'] - perf_stats['sim_time'] - perf_stats['nn_time'] - perf_stats['mc_equity_time'])
    
    return {
        'transitions': batch_transitions,
        'hands_completed': hands_to_play,
        'action_counts': action_counts,
        'position_history': position_history,
        'hand_profits': hand_profits,
        'vpip_count': vpip_count,
        'pfr_count': pfr_count,
        'three_bet_count': three_bet_count,
        'perf_stats': perf_stats,
    }
