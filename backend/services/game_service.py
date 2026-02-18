"""
Game service for managing poker game state and building model observations.

Converts frontend game state to the observation format expected by the trained models.
V19+: Uses range-weighted Monte Carlo equity for more realistic post-flop estimates.
"""

import math
import random
from itertools import combinations
from typing import Optional

import numpy as np
from pokerkit import Card, Deck, StandardHighHand

from backend.config import get_settings
from backend.models.schemas import (
    GameStateRequest, 
    CardSchema, 
    HAND_STRENGTH_CATEGORIES,
    ACTION_NAMES,
)


# Action constants
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE_SMALL = 2
ACTION_RAISE_MEDIUM = 3
ACTION_RAISE_LARGE = 4
ACTION_ALL_IN = 5

NUM_ACTIONS = 6


# =============================================================================
# Hand Strength Rankings for V19 Weighted Equity (0=AA best, 100=72o worst)
# =============================================================================

HAND_RANK_TABLE = {
    "AAo": 0,  "KKo": 1,  "QQo": 2,  "JJo": 3,  "AKs": 4,
    "AQs": 5,  "TTo": 6,  "AKo": 7,  "AJs": 8,  "KQs": 9,
    "99o": 10, "ATs": 11, "KJs": 12, "AQo": 13, "QJs": 14,
    "KTs": 15, "88o": 16, "A9s": 17, "QTs": 18, "AJo": 19,
    "JTs": 20, "77o": 21, "KQo": 22, "A8s": 23, "K9s": 24,
    "ATo": 25, "Q9s": 26, "T9s": 27, "J9s": 28, "66o": 29,
    "A7s": 30, "A5s": 31, "KJo": 32, "55o": 33, "A6s": 34,
    "K8s": 35, "QJo": 36, "A4s": 37, "KTo": 38, "98s": 39,
    "Q8s": 40, "T8s": 41, "A3s": 42, "J8s": 43, "44o": 44,
    "QTo": 45, "A2s": 46, "K7s": 47, "87s": 48, "JTo": 49,
    "K6s": 50, "97s": 51, "33o": 52, "76s": 53, "Q7s": 54,
    "T9o": 55, "K5s": 56, "J7s": 57, "T7s": 58, "22o": 59,
    "Q6s": 59, "K4s": 60, "86s": 61, "65s": 62, "J9o": 63,
    "K3s": 64, "Q5s": 65, "96s": 66, "54s": 67, "K2s": 68,
    "Q4s": 69, "75s": 70, "98o": 71, "Q3s": 72, "J6s": 73,
    "T6s": 74, "85s": 75, "64s": 76, "Q2s": 77, "J5s": 78,
    "53s": 79, "T8o": 80, "87o": 81, "J4s": 82, "95s": 83,
    "74s": 84, "43s": 85, "J3s": 86, "J2s": 87, "97o": 88,
    "T5s": 89, "84s": 90, "63s": 91, "76o": 92, "T4s": 93,
    "52s": 94, "T3s": 95, "86o": 96, "T2s": 97, "65o": 98,
    "94s": 99, "42s": 100, "93s": 100, "73s": 100, "54o": 100,
    "92s": 100, "96o": 100, "83s": 100, "32s": 100, "85o": 100,
    "75o": 100, "82s": 100, "62s": 100, "72s": 100, "95o": 100,
    "64o": 100, "53o": 100, "74o": 100, "43o": 100, "84o": 100,
    "T9o": 100, "93o": 100, "63o": 100, "94o": 100, "52o": 100,
    "73o": 100, "42o": 100, "92o": 100, "83o": 100, "32o": 100,
    "82o": 100, "62o": 100, "72o": 100,
}

RANK_ORDER = "AKQJT98765432"


def get_hand_rank(card1: Card, card2: Card) -> int:
    """Look up starting hand rank (0=AA best, 100=72o worst)."""
    rank1 = card1.rank
    rank2 = card2.rank
    suited = (card1.suit == card2.suit)
    
    idx1 = RANK_ORDER.index(rank1)
    idx2 = RANK_ORDER.index(rank2)
    if idx1 > idx2:
        rank1, rank2 = rank2, rank1
    
    if rank1 == rank2:
        key = f"{rank1}{rank2}o"
    elif suited:
        key = f"{rank1}{rank2}s"
    else:
        key = f"{rank1}{rank2}o"
    
    return HAND_RANK_TABLE.get(key, 100)


def hand_weight(rank: int, street: int) -> float:
    """
    Convert hand rank + street into a sampling weight [0, 1].
    Preflop: all 1.0. Post-flop: sigmoid filtering, tighter on later streets.
    """
    if street == 0:
        return 1.0
    
    if street == 1:  # Flop
        cutoff, steepness, min_weight = 60.0, 0.08, 0.10
    elif street == 2:  # Turn
        cutoff, steepness, min_weight = 45.0, 0.10, 0.05
    else:  # River
        cutoff, steepness, min_weight = 35.0, 0.12, 0.03
    
    weight = 1.0 / (1.0 + math.exp(steepness * (rank - cutoff)))
    return max(weight, min_weight)


def get_street_from_board(board_cards: list) -> int:
    """Determine current street from board card count."""
    n = len(board_cards)
    if n == 0: return 0    # Preflop
    elif n == 3: return 1  # Flop
    elif n == 4: return 2  # Turn
    else: return 3         # River


def card_schema_to_pokerkit(card: CardSchema) -> Card:
    """Convert a CardSchema to a pokerkit Card."""
    # Normalize rank (handle both 'T' and '10')
    rank = card.rank.upper()
    if rank == "10":
        rank = "T"
    
    # pokerkit uses lowercase suits
    suit = card.suit.lower()
    
    # pokerkit Card.parse returns an iterator, so we take the first item
    return next(Card.parse(rank + suit))


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


class GameService:
    """
    Service for converting frontend game state to model observations
    and determining legal actions.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def monte_carlo_equity(
        self, 
        hole_cards: list[Card], 
        board_cards: list[Card],
        num_opponents: int = 1,
        iterations: int = None,
        street: int = 0
    ) -> float:
        """
        Calculate hand equity via Monte Carlo simulation.
        
        V19+: Uses range-weighted opponent sampling on post-flop streets.
        Opponent hands are filtered via rejection sampling based on their
        starting hand rank, so weak hands that realistic opponents would
        have folded are less likely to appear.
        
        Args:
            hole_cards: Bot's hole cards
            board_cards: Community cards
            num_opponents: Number of active opponents
            iterations: Number of simulations (uses config default if not specified)
            street: 0=preflop, 1=flop, 2=turn, 3=river (controls filtering)
            
        Returns:
            Equity as a float between 0 and 1
        """
        iterations = iterations or self.settings.equity_iterations
        
        if not hole_cards:
            return 0.5
        
        wins = 0
        valid_iterations = 0
        known_cards = set(hole_cards + board_cards)
        deck_cards = [c for c in Deck.STANDARD if c not in known_cards]
        needed_board = 5 - len(board_cards)
        
        # More attempts when filtering, since some will be rejected
        max_attempts = iterations * 4 if street > 0 else iterations
        attempts = 0
        
        while valid_iterations < iterations and attempts < max_attempts:
            attempts += 1
            random.shuffle(deck_cards)
            
            idx = 0
            opponent_hands = []
            rejected = False
            
            for _ in range(num_opponents):
                opp_hole = deck_cards[idx:idx + 2]
                idx += 2
                
                # Rejection sampling based on hand strength (V19)
                if street > 0 and len(opp_hole) == 2:
                    rank = get_hand_rank(opp_hole[0], opp_hole[1])
                    weight = hand_weight(rank, street)
                    if random.random() > weight:
                        rejected = True
                        break
                
                opponent_hands.append(opp_hole)
            
            if rejected:
                continue
            
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
            
            valid_iterations += 1
        
        if valid_iterations == 0:
            return 0.5  # Fallback if all samples were rejected
        
        return wins / valid_iterations
    
    def build_observation(self, game_state: GameStateRequest) -> tuple[np.ndarray, float]:
        """
        Build the observation vector from game state.
        
        Args:
            game_state: The game state from the frontend
            
        Returns:
            Tuple of (observation array, equity)
        """
        obs = []
        
        # Get bot's player state
        bot_player = None
        for player in game_state.players:
            if player.is_bot:
                bot_player = player
                break
        
        if bot_player is None:
            raise ValueError("No bot player found in game state")
        
        if bot_player.hole_cards is None:
            raise ValueError("Bot player must have hole cards")
        
        # Convert cards
        hole_cards = [card_schema_to_pokerkit(c) for c in bot_player.hole_cards]
        board_cards = [card_schema_to_pokerkit(c) for c in game_state.community_cards]
        
        # Card encoding
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        
        # Encode hole cards (2 x 52)
        for i in range(2):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(hole_cards):
                idx = ranks.index(hole_cards[i].rank) * 4 + suits.index(hole_cards[i].suit)
                encoding[idx] = 1.0
            obs.extend(encoding)
        
        # Encode board cards (5 x 52)
        for i in range(5):
            encoding = np.zeros(52, dtype=np.float32)
            if i < len(board_cards):
                idx = ranks.index(board_cards[i].rank) * 4 + suits.index(board_cards[i].suit)
                encoding[idx] = 1.0
            obs.extend(encoding)
        
        # Calculate equity (V19: street-aware weighted sampling)
        num_opponents = sum(1 for p in game_state.players if p.is_active and not p.is_bot)
        current_street = get_street_from_board(board_cards)
        equity = self.monte_carlo_equity(
            hole_cards, board_cards, max(1, num_opponents), street=current_street
        )
        
        # Stack and pot info
        total_pot = game_state.pot
        current_bet = game_state.current_bet
        my_bet = bot_player.bet
        to_call = current_bet - my_bet
        my_stack = bot_player.stack
        big_blind = game_state.big_blind
        starting_stack = self.settings.default_starting_stack
        
        # Get opponent stacks
        opp_stacks = [p.stack for p in game_state.players if not p.is_bot]
        num_players = len(game_state.players)
        
        obs.append(equity)
        
        pot_odds = to_call / (total_pot + to_call + 1e-6)
        obs.append(pot_odds)
        obs.append(min(to_call / (total_pot + 1e-6), 2.0))
        obs.append(min((my_stack / (total_pot + 1e-6)) / 20.0, 1.0))
        obs.append(my_stack / starting_stack)
        obs.append(total_pot / (starting_stack * num_players))
        
        # Opponent stacks (pad to 5 opponents for 6-max)
        for opp_stack in opp_stacks[:5]:
            obs.append(opp_stack / 1000.0)
        while len(opp_stacks) < 5:
            obs.append(0.0)
            opp_stacks.append(0)  # Just for padding count
        
        active_opponents = sum(1 for p in game_state.players if p.is_active and not p.is_bot)
        obs.append(active_opponents / 5.0)
        
        breakeven_equity = to_call / (total_pot + to_call + 1e-6)
        obs.append(breakeven_equity)
        obs.append(equity - breakeven_equity)
        obs.append((starting_stack - my_stack) / starting_stack)
        obs.append(1.0 if to_call > 0 else 0.0)
        
        # Hand strength category (one-hot)
        strength_cat = compute_hand_strength_category(equity)
        for i in range(5):
            obs.append(1.0 if i == strength_cat else 0.0)
        
        # Street encoding
        street = [0.0] * 4
        num_board = len(board_cards)
        if num_board == 0:
            street[0] = 1.0  # Preflop
        elif num_board == 3:
            street[1] = 1.0  # Flop
        elif num_board == 4:
            street[2] = 1.0  # Turn
        else:
            street[3] = 1.0  # River
        obs.extend(street)
        
        # Position encoding (6-max)
        position_encoding = [0.0] * 6
        position_encoding[bot_player.position % 6] = 1.0
        obs.extend(position_encoding)
        
        # Players after/before
        players_after = sum(1 for p in game_state.players 
                          if p.position > bot_player.position and p.is_active)
        obs.append(players_after / 5.0)
        
        players_before = sum(1 for p in game_state.players 
                           if p.position < bot_player.position and p.is_active)
        obs.append(players_before / 5.0)
        
        # Pad to 520 dimensions
        while len(obs) < 520:
            obs.append(0.0)
        
        return np.array(obs[:520], dtype=np.float32), equity
    
    def get_legal_actions(self, game_state: GameStateRequest) -> list[int]:
        """
        Determine legal actions based on game state.
        
        In IRL mode, we infer legal actions from the betting state.
        """
        legal = []
        
        # Get bot's player state
        bot_player = None
        for player in game_state.players:
            if player.is_bot:
                bot_player = player
                break
        
        if bot_player is None:
            return [ACTION_CALL]  # Fallback
        
        to_call = game_state.current_bet - bot_player.bet
        
        # Can always fold if there's something to call
        if to_call > 0:
            legal.append(ACTION_FOLD)
        
        # Can always check/call
        legal.append(ACTION_CALL)
        
        # Can raise if has chips
        if bot_player.stack > to_call:
            legal.append(ACTION_RAISE_SMALL)
            legal.append(ACTION_RAISE_MEDIUM)
            legal.append(ACTION_RAISE_LARGE)
            legal.append(ACTION_ALL_IN)
        elif bot_player.stack > 0:
            # Can only go all-in
            legal.append(ACTION_ALL_IN)
        
        return legal if legal else [ACTION_CALL]
    
    def calculate_raise_amount(
        self, 
        action: int, 
        game_state: GameStateRequest
    ) -> Optional[int]:
        """
        Calculate the chip amount for a raise action.
        
        Args:
            action: The action ID
            game_state: Current game state
            
        Returns:
            Chip amount for raises, None for fold/call
        """
        if action in [ACTION_FOLD, ACTION_CALL]:
            return None
        
        bot_player = None
        for player in game_state.players:
            if player.is_bot:
                bot_player = player
                break
        
        if bot_player is None:
            return None
        
        big_blind = game_state.big_blind
        pot = game_state.pot
        stack = bot_player.stack
        current_bet = game_state.current_bet
        my_bet = bot_player.bet
        to_call = current_bet - my_bet
        
        # Minimum raise is typically 2x the current bet or big blind
        min_raise = max(current_bet * 2, big_blind * 2)
        max_raise = stack + my_bet  # All-in amount
        
        if action == ACTION_RAISE_SMALL:
            # 2x min raise
            amount = min(min_raise * 2, max_raise)
        elif action == ACTION_RAISE_MEDIUM:
            # 0.5x pot
            amount = min(max(min_raise, int(pot * 0.5) + current_bet), max_raise)
        elif action == ACTION_RAISE_LARGE:
            # 1x pot
            amount = min(max(min_raise, pot + current_bet), max_raise)
        elif action == ACTION_ALL_IN:
            amount = max_raise
        else:
            return None
        
        return int(amount)


# Singleton instance
_game_service: Optional[GameService] = None


def get_game_service() -> GameService:
    """Get the singleton game service instance."""
    global _game_service
    if _game_service is None:
        _game_service = GameService()
    return _game_service
