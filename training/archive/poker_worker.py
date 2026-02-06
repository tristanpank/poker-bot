"""
Poker Episode Worker Module for V12

This module contains the episode runner function that can be used with multiprocessing.
It must be in a separate .py file for spawn-based multiprocessing to work in Jupyter.
"""

import random
from itertools import combinations
from typing import List, Dict, Any
import numpy as np
import torch

from pokerkit import Automation, NoLimitTexasHoldem, Card, StandardHighHand, Deck
from poker_model import StatisticsPokerNet


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


def monte_carlo_equity_fast(hole_cards: List[Card], board_cards: List[Card], 
                            iterations: int = 20) -> float:
    """Optimized equity calculation."""
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


def run_single_episode(args):
    """
    Run a single poker episode. Returns all transitions and stats.
    This function runs in a separate process.
    """
    episode_seed, epsilon, opp_type, equity_iterations, model_state_dict = args
    random.seed(episode_seed)
    
    # Load model for action selection
    model = StatisticsPokerNet()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Create environment
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
        state_vector = []
        hole = flatten_cards_list(state.hole_cards[0])
        board = flatten_cards_list(state.board_cards)
        
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        
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
        my_bet = state.bets[0]
        to_call = current_bet - my_bet
        my_stack = state.stacks[0]
        opp_stack = state.stacks[1]
        
        equity = monte_carlo_equity_fast(hole, board, equity_iterations)
        state_vector.append(equity)
        
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(pot_odds)
        state_vector.append(min(to_call / (total_pot + 1e-6), 2.0))
        state_vector.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
        state_vector.append(min(my_stack, opp_stack) / starting_stack)
        state_vector.append(my_stack / starting_stack)
        state_vector.append(opp_stack / starting_stack)
        state_vector.append(total_pot / (starting_stack * 2))
        
        breakeven_equity = to_call / (total_pot + to_call + 1e-6)
        state_vector.append(breakeven_equity)
        excess_equity = equity - breakeven_equity
        state_vector.append(excess_equity)
        state_vector.append((starting_stack - my_stack) / starting_stack)
        state_vector.append(1.0 if to_call > 0 else 0.0)
        
        street = [0.0, 0.0, 0.0, 0.0]
        if len(board) == 0: street[0] = 1.0
        elif len(board) == 3: street[1] = 1.0
        elif len(board) == 4: street[2] = 1.0
        else: street[3] = 1.0
        state_vector.extend(street)
        
        return np.array(state_vector, dtype=np.float32), {
            'equity': equity, 'pot_bb': total_pot / big_blind,
            'to_call_bb': to_call / big_blind, 'breakeven_equity': breakeven_equity,
            'excess_equity': excess_equity
        }
    
    def get_legal_actions():
        legal = []
        if state.can_fold(): legal.append(0)
        if state.can_check_or_call(): legal.append(1)
        if state.can_complete_bet_or_raise_to(): legal.append(2)
        return legal if legal else [1]
    
    def execute_action(action):
        if action == 0:
            if state.can_fold(): state.fold()
            elif state.can_check_or_call(): state.check_or_call()
        elif action == 1:
            if state.can_check_or_call(): state.check_or_call()
            elif state.can_fold(): state.fold()
        elif action == 2:
            if state.can_complete_bet_or_raise_to():
                min_r = state.min_completion_betting_or_raising_to_amount
                max_r = state.max_completion_betting_or_raising_to_amount
                state.complete_bet_or_raise_to(min(min_r * 2, max_r))
            elif state.can_check_or_call(): state.check_or_call()
    
    def opponent_action(legal):
        equity = monte_carlo_equity_fast(
            flatten_cards_list(state.hole_cards[1]),
            flatten_cards_list(state.board_cards),
            equity_iterations
        )
        
        if opp_type == 'value':
            if equity > 0.75 and 2 in legal: return 2
            if equity > 0.50 and 1 in legal: return 1
            if 0 in legal: return 0
            return 1
        elif opp_type == 'bluff':
            if equity > 0.70 and 2 in legal: return 2
            if equity < 0.40 and random.random() < 0.35 and 2 in legal: return 2
            if equity > 0.45 and 1 in legal: return 1
            if 0 in legal: return 0
            return 1
        else:  # balanced
            if equity > 0.80 and 2 in legal: return 2
            if 1 in legal:
                if equity > 0.60: return 1
                to_call = max(state.bets) - state.bets[1]
                pot_odds = to_call / (sum(state.bets) + to_call + 1e-5)
                if equity > pot_odds + 0.05: return 1
            if 0 in legal: return 0
            return 1
    
    # Play episode
    transitions = []
    contexts = []
    actions_taken = []
    action_counts = {0: 0, 1: 0, 2: 0}
    
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
            
            # Epsilon-greedy action selection using trained model
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
            
            execute_action(action)
            run_automations()
        else:  # Opponent
            legal = get_legal_actions()
            action = opponent_action(legal)
            execute_action(action)
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
