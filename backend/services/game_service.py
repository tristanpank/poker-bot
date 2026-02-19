"""
Game service for managing poker game state and building model observations.

Converts frontend game state to the observation format expected by the trained models.
"""

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
        iterations: int = None
    ) -> float:
        """
        Calculate hand equity via Monte Carlo simulation.
        
        Args:
            hole_cards: Bot's hole cards
            board_cards: Community cards
            num_opponents: Number of active opponents
            iterations: Number of simulations (uses config default if not specified)
            
        Returns:
            Equity as a float between 0 and 1
        """
        iterations = iterations or self.settings.equity_iterations
        
        if not hole_cards:
            return 0.5
        
        wins = 0
        known_cards = set(hole_cards + board_cards)
        deck_cards = [c for c in Deck.STANDARD if c not in known_cards]
        needed_board = 5 - len(board_cards)
        
        for _ in range(iterations):
            random.shuffle(deck_cards)
            
            idx = 0
            opponent_hands = []
            for _ in range(num_opponents):
                opp_hole = deck_cards[idx:idx + 2]
                opponent_hands.append(opp_hole)
                idx += 2
            
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
        
        # Calculate equity
        num_opponents = sum(1 for p in game_state.players if p.is_active and not p.is_bot)
        equity = self.monte_carlo_equity(hole_cards, board_cards, max(1, num_opponents))
        
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
