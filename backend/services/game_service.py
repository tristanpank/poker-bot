"""
Game service for managing poker game state and building model observations.

Converts frontend game state to the observation format expected by the trained models.
V19+: Uses range-weighted Monte Carlo equity for more realistic post-flop estimates.
"""

import math
import random
from itertools import combinations
from typing import Literal, Optional

import numpy as np
from pokerkit import Card, Deck, StandardHighHand

from backend.config import get_settings
from backend.models.schemas import (
    GameStateRequest,
    CardSchema,
    PlayerState,
)


# Action constants (v21)
ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_HALF_POT = 3
ACTION_RAISE_POT_OR_ALL_IN = 4

NUM_ACTIONS = 5

FRONTEND_ACTION_FOLD = "fold"
FRONTEND_ACTION_CHECK = "check"
FRONTEND_ACTION_CALL = "call"
FRONTEND_ACTION_RAISE = "raise_amt"

FEATURE_RANKS = "23456789TJQKA"
FEATURE_SUITS = "cdhs"


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


def _rank_index(card: Card) -> int:
    rank = getattr(card.rank, "value", card.rank)
    return FEATURE_RANKS.index(rank)


def _suit_index(card: Card) -> int:
    suit = getattr(card.suit, "value", card.suit)
    return FEATURE_SUITS.index(suit)


def _normalize(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    clipped = max(0.0, min(cap, value))
    return float(clipped / cap)


def _estimate_preflop_strength(hole_cards: list[Card], num_opponents: int = 1) -> float:
    if len(hole_cards) != 2:
        return 0.35

    r1 = _rank_index(hole_cards[0])
    r2 = _rank_index(hole_cards[1])
    hi = max(r1, r2) / 12.0
    lo = min(r1, r2) / 12.0
    pair = 1.0 if r1 == r2 else 0.0
    suited = 1.0 if _suit_index(hole_cards[0]) == _suit_index(hole_cards[1]) else 0.0
    gap = abs(r1 - r2)
    connected = 1.0 if gap == 1 else 0.0
    one_gap = 1.0 if gap == 2 else 0.0
    broadway = 1.0 if max(r1, r2) >= 8 and min(r1, r2) >= 7 else 0.0

    base = (
        0.08
        + (0.40 * hi)
        + (0.16 * lo)
        + (0.22 * pair)
        + (0.08 * suited)
        + (0.04 * connected)
        + (0.02 * one_gap)
        + (0.03 * broadway)
    )
    opponent_penalty = max(0, num_opponents - 1) * 0.035
    return float(max(0.0, min(0.99, base - opponent_penalty)))


def _board_connected(board_cards: list[Card]) -> float:
    if len(board_cards) < 3:
        return 0.0
    ranks = sorted(set(_rank_index(card) for card in board_cards))
    if 12 in ranks:
        ranks = sorted(set(ranks + [-1]))
    for start in range(len(ranks)):
        window = ranks[start:start + 3]
        if len(window) == 3 and (window[-1] - window[0]) <= 4:
            return 1.0
    return 0.0


def _has_flush_draw(cards: list[Card]) -> float:
    suit_counts = [0, 0, 0, 0]
    for card in cards:
        suit_counts[_suit_index(card)] += 1
    return 1.0 if max(suit_counts, default=0) >= 4 else 0.0


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
    
    def _get_bot_player(self, game_state: GameStateRequest):
        for player in game_state.players:
            if player.is_bot:
                return player
        return None

    def _get_raise_bounds(self, game_state: GameStateRequest, acting_player) -> tuple[int, int]:
        current_bet = max(0, int(game_state.current_bet))
        hero_bet = max(0, int(acting_player.bet))
        stack = max(0, int(acting_player.stack))
        big_blind = max(1, int(game_state.big_blind))
        to_call = max(0, current_bet - hero_bet)

        max_raise_to = hero_bet + stack
        if stack <= to_call:
            return 0, max_raise_to

        # Approximate minimum legal no-limit raise sizing from available UI state.
        min_increment = max(big_blind, to_call)
        min_raise_to = current_bet + min_increment
        if current_bet <= 0:
            min_raise_to = max(big_blind, hero_bet + big_blind)

        return int(min_raise_to), int(max_raise_to)

    def _next_active_player(self, players, after_idx: int) -> int:
        n = len(players)
        for offset in range(1, n + 1):
            idx = (after_idx + offset) % n
            if players[idx].is_active:
                return idx
        return -1

    def _get_legal_action_info_for_actor(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> dict[str, object]:
        if actor_index < 0 or actor_index >= len(game_state.players):
            return {
                "actor_index": -1,
                "actions": [],
                "to_call": 0,
                "min_raise_to": None,
                "max_raise_to": None,
            }

        actor = game_state.players[actor_index]
        if not actor.is_active:
            return {
                "actor_index": actor_index,
                "actions": [],
                "to_call": 0,
                "min_raise_to": None,
                "max_raise_to": None,
            }

        current_bet = max(0, int(game_state.current_bet))
        actor_bet = max(0, int(actor.bet))
        stack = max(0, int(actor.stack))
        to_call = max(0, current_bet - actor_bet)

        actions: list[Literal["fold", "check", "call", "raise_amt"]] = []
        if to_call > 0:
            actions.append(FRONTEND_ACTION_FOLD)
        if to_call == 0:
            actions.append(FRONTEND_ACTION_CHECK)
        elif stack > 0:
            actions.append(FRONTEND_ACTION_CALL)

        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, actor)
        can_raise = (
            stack > to_call
            and max_raise_to >= min_raise_to
            and max_raise_to > current_bet
        )
        if can_raise:
            actions.append(FRONTEND_ACTION_RAISE)
        else:
            min_raise_to = 0
            max_raise_to = 0

        return {
            "actor_index": actor_index,
            "actions": actions,
            "to_call": to_call,
            "min_raise_to": int(min_raise_to) if can_raise else None,
            "max_raise_to": int(max_raise_to) if can_raise else None,
        }

    def get_legal_action_info(
        self,
        game_state: GameStateRequest,
        actor_index: Optional[int] = None,
    ) -> dict[str, object]:
        idx = game_state.current_player_idx if actor_index is None else actor_index
        return self._get_legal_action_info_for_actor(game_state, int(idx))

    def get_legal_action_ids_for_actor(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> list[int]:
        info = self._get_legal_action_info_for_actor(game_state, actor_index)
        action_ids: list[int] = []
        for action in info["actions"]:
            if action == FRONTEND_ACTION_FOLD:
                action_ids.append(ACTION_FOLD)
            elif action == FRONTEND_ACTION_CHECK:
                action_ids.append(ACTION_CHECK)
            elif action == FRONTEND_ACTION_CALL:
                action_ids.append(ACTION_CALL)
            elif action == FRONTEND_ACTION_RAISE:
                action_ids.extend([ACTION_RAISE_HALF_POT, ACTION_RAISE_POT_OR_ALL_IN])
        return sorted(set(action_ids))

    def model_action_to_frontend(self, action_id: int) -> Literal["fold", "check", "call", "raise_amt"]:
        if action_id == ACTION_FOLD:
            return FRONTEND_ACTION_FOLD
        if action_id == ACTION_CHECK:
            return FRONTEND_ACTION_CHECK
        if action_id == ACTION_CALL:
            return FRONTEND_ACTION_CALL
        return FRONTEND_ACTION_RAISE

    def apply_frontend_action(
        self,
        game_state: GameStateRequest,
        actor_index: int,
        action: Literal["fold", "check", "call", "raise_amt"],
        raise_amt: Optional[int] = None,
    ) -> tuple[GameStateRequest, bool, Optional[int]]:
        if actor_index < 0 or actor_index >= len(game_state.players):
            raise ValueError("Invalid actor index")
        if game_state.current_player_idx != actor_index:
            raise ValueError("Action actor does not match current_player_idx")

        legal = self._get_legal_action_info_for_actor(game_state, actor_index)
        legal_actions = set(legal["actions"])
        if action not in legal_actions:
            raise ValueError(f"Illegal action '{action}' for actor index {actor_index}")

        next_state = game_state.model_copy(deep=True)
        players = next_state.players
        actor = players[actor_index]
        current_bet = max(0, int(next_state.current_bet))
        pot = max(0, int(next_state.pot))
        to_call = int(legal["to_call"])
        applied_raise_amt: Optional[int] = None

        if action == FRONTEND_ACTION_FOLD:
            actor.is_active = False
            actor.has_acted = True
        elif action == FRONTEND_ACTION_CHECK:
            actor.has_acted = True
        elif action == FRONTEND_ACTION_CALL:
            call_amt = min(to_call, max(0, int(actor.stack)))
            actor.bet += call_amt
            actor.stack -= call_amt
            actor.has_acted = True
            pot += call_amt
        elif action == FRONTEND_ACTION_RAISE:
            min_raise_to = legal["min_raise_to"]
            max_raise_to = legal["max_raise_to"]
            if min_raise_to is None or max_raise_to is None:
                raise ValueError("Raise is not legal in this state")

            target = int(min_raise_to if raise_amt is None else raise_amt)
            if target < int(min_raise_to) or target > int(max_raise_to):
                raise ValueError(f"Raise amount must be within [{min_raise_to}, {max_raise_to}]")

            additional = target - int(actor.bet)
            if additional <= 0:
                raise ValueError("Raise amount must exceed current bet")
            if additional > int(actor.stack):
                raise ValueError("Raise amount exceeds actor stack")

            # Reopen action: other active players must act again.
            for idx, player in enumerate(players):
                if idx != actor_index and player.is_active:
                    player.has_acted = False

            actor.bet = target
            actor.stack -= additional
            actor.has_acted = True
            pot += additional
            current_bet = target
            applied_raise_amt = target

        next_state.pot = pot
        next_state.current_bet = current_bet

        active_players = [p for p in players if p.is_active]
        all_acted_and_matched = all(
            p.has_acted and (p.bet == current_bet or p.stack == 0)
            for p in active_players
        )
        round_complete = (len(active_players) <= 1) or all_acted_and_matched

        if round_complete:
            next_state.current_player_idx = -1
        else:
            next_idx = self._next_active_player(players, actor_index)
            if next_idx == -1:
                next_state.current_player_idx = -1
                round_complete = True
            else:
                next_state.current_player_idx = next_idx

        return next_state, round_complete, applied_raise_amt

    def _best_hand_rank(self, hole_cards: list[Card], board_cards: list[Card]) -> StandardHighHand:
        if len(hole_cards) != 2:
            raise ValueError("Expected exactly two hole cards")
        if len(board_cards) != 5:
            raise ValueError("Expected exactly five community cards for showdown")
        combined = hole_cards + board_cards
        return max(StandardHighHand(combo) for combo in combinations(combined, 5))

    def resolve_hand_result(
        self,
        game_state: GameStateRequest,
        starting_stacks: list[int],
        opponents: dict[int, dict[str, object]],
    ) -> dict[str, object]:
        players = game_state.players
        n_players = len(players)
        if len(starting_stacks) != n_players:
            raise ValueError("starting_stacks length must match players length")

        bot_index = next((idx for idx, player in enumerate(players) if player.is_bot), -1)
        if bot_index < 0:
            raise ValueError("No bot player found")

        bot_start = int(starting_stacks[bot_index])
        bot_stack = int(players[bot_index].stack)
        if bot_start < bot_stack:
            raise ValueError("starting_stacks cannot be lower than current stacks")
        bot_contribution = bot_start - bot_stack
        pot = int(max(0, game_state.pot))

        for i, player in enumerate(players):
            if int(starting_stacks[i]) < int(player.stack):
                raise ValueError("starting_stacks cannot be lower than current stacks")

        active_indices = [idx for idx, player in enumerate(players) if player.is_active]
        if not active_indices:
            raise ValueError("Cannot resolve hand: no active players")

        if len(active_indices) == 1:
            winner_indices = [active_indices[0]]
        else:
            board_cards = [card_schema_to_pokerkit(card) for card in game_state.community_cards]
            if len(board_cards) != 5:
                raise ValueError("Showdown requires 5 community cards when multiple active players remain")

            seen: set[str] = set()
            for card in game_state.community_cards:
                card_id = f"{card.rank.upper()}{card.suit.lower()}"
                if card_id in seen:
                    raise ValueError("Duplicate board card detected")
                seen.add(card_id)

            contender_cards: dict[int, list[Card]] = {}
            showdown_contenders: list[int] = []
            for idx in active_indices:
                player = players[idx]
                info = opponents.get(idx, {})
                mucked = bool(info.get("mucked", False))
                if mucked:
                    continue

                if player.is_bot:
                    if player.hole_cards is None or len(player.hole_cards) != 2:
                        raise ValueError("Bot hole cards are required to resolve showdown")
                    cards = player.hole_cards
                else:
                    cards = info.get("hole_cards")
                    if cards is None:
                        raise ValueError(f"Missing revealed hole cards for active opponent index {idx}")
                    if len(cards) != 2:
                        raise ValueError(f"Opponent index {idx} must have exactly 2 revealed cards")

                hole_cards = [card_schema_to_pokerkit(card) for card in cards]
                for card in cards:
                    card_id = f"{card.rank.upper()}{card.suit.lower()}"
                    if card_id in seen:
                        raise ValueError("Duplicate card detected between board and hole cards")
                    seen.add(card_id)

                contender_cards[idx] = hole_cards
                showdown_contenders.append(idx)

            if not showdown_contenders:
                winner_indices = []
            elif len(showdown_contenders) == 1:
                winner_indices = showdown_contenders
            else:
                ranked = [
                    (idx, self._best_hand_rank(hole_cards, board_cards))
                    for idx, hole_cards in contender_cards.items()
                ]
                best_rank = max(rank for _, rank in ranked)
                winner_indices = sorted(idx for idx, rank in ranked if rank == best_rank)

        payouts = [0] * n_players
        if winner_indices:
            share = pot // len(winner_indices)
            remainder = pot % len(winner_indices)
            for idx in winner_indices:
                payouts[idx] += share
            for idx in winner_indices:
                if remainder <= 0:
                    break
                payouts[idx] += 1
                remainder -= 1

        bot_payout = int(payouts[bot_index])
        delta = int(bot_payout - bot_contribution)
        if delta > 0:
            result = "won"
        elif delta < 0:
            result = "lost"
        else:
            result = "push"

        final_stacks = [
            max(0, int(player.stack) + int(payouts[idx]))
            for idx, player in enumerate(players)
        ]

        ordered_players = sorted(
            list(enumerate(players)),
            key=lambda item: int(item[1].position),
        )
        if not ordered_players:
            raise ValueError("Cannot build next hand state without players")
        rotated_players = ordered_players[1:] + ordered_players[:1]

        next_players: list[PlayerState] = []
        next_bot_position = 0
        for new_position, (old_index, old_player) in enumerate(rotated_players):
            stack = final_stacks[old_index]
            is_active = stack > 0
            next_player = PlayerState(
                position=new_position,
                stack=stack,
                bet=0,
                hole_cards=[] if old_player.is_bot else None,
                is_bot=old_player.is_bot,
                is_active=is_active,
                has_acted=False,
            )
            next_players.append(next_player)
            if old_player.is_bot:
                next_bot_position = new_position

        next_actor_idx = -1
        for idx, player in enumerate(next_players):
            if player.is_active:
                next_actor_idx = idx
                break

        next_game_state = GameStateRequest(
            session_id=game_state.session_id,
            community_cards=[],
            pot=0,
            players=next_players,
            bot_position=next_bot_position,
            current_bet=0,
            big_blind=int(game_state.big_blind),
            current_player_idx=next_actor_idx,
            model_version=game_state.model_version,
        )

        return {
            "result": result,
            "amount": abs(delta),
            "delta": delta,
            "bot_payout": bot_payout,
            "bot_contribution": bot_contribution,
            "pot": pot,
            "winner_indices": winner_indices,
            "next_game_state": next_game_state,
        }

    def build_observation(self, game_state: GameStateRequest) -> tuple[np.ndarray, float]:
        """
        Build the 98-dim v21 observation vector and equity estimate.
        """
        bot_player = self._get_bot_player(game_state)
        if bot_player is None:
            raise ValueError("No bot player found in game state")
        if bot_player.hole_cards is None:
            raise ValueError("Bot player must have hole cards")

        hole_cards = [card_schema_to_pokerkit(c) for c in bot_player.hole_cards]
        board_cards = [card_schema_to_pokerkit(c) for c in game_state.community_cards]
        board_len = len(board_cards)
        street_idx = get_street_from_board(board_cards)

        num_opponents = sum(1 for p in game_state.players if p.is_active and not p.is_bot)
        equity = self.monte_carlo_equity(
            hole_cards,
            board_cards,
            max(1, num_opponents),
            street=street_idx,
        )

        obs = np.zeros(98, dtype=np.float32)

        if len(hole_cards) == 2:
            rank_indices = sorted((_rank_index(card) for card in hole_cards))
            low_rank = rank_indices[0]
            high_rank = rank_indices[1]
            obs[high_rank] = 1.0
            obs[13 + low_rank] = 1.0
            obs[26] = 1.0 if _suit_index(hole_cards[0]) == _suit_index(hole_cards[1]) else 0.0
            obs[27] = 1.0 if low_rank == high_rank else 0.0

            gap = high_rank - low_rank
            if high_rank == 12 and low_rank == 0:
                gap = 1
            if gap == 1:
                obs[28] = 1.0
            elif gap == 2:
                obs[29] = 1.0
            elif gap == 3:
                obs[30] = 1.0
            else:
                obs[31] = 1.0

            obs[96] = float(high_rank / 12.0)
            obs[97] = float(low_rank / 12.0)

        obs[49 + street_idx] = 1.0

        board_rank_counts = np.zeros(13, dtype=np.int32)
        board_suit_counts = np.zeros(4, dtype=np.int32)
        for card in board_cards:
            board_rank_counts[_rank_index(card)] += 1
            board_suit_counts[_suit_index(card)] += 1

        if board_len > 0:
            inv_len = 1.0 / float(board_len)
            obs[32:45] = board_rank_counts.astype(np.float32) * inv_len
            obs[45:49] = board_suit_counts.astype(np.float32) * inv_len

        obs[53] = 1.0 if np.any(board_rank_counts >= 2) else 0.0
        obs[54] = 1.0 if np.any(board_rank_counts >= 3) else 0.0
        obs[55] = 1.0 if board_len >= 3 and np.any(board_suit_counts == board_len) else 0.0
        obs[56] = 1.0 if board_len >= 3 and np.count_nonzero(board_suit_counts) == 2 else 0.0
        obs[57] = _board_connected(board_cards)

        active_opponents = max(1, sum(1 for p in game_state.players if p.is_active and not p.is_bot))
        obs[58] = _estimate_preflop_strength(hole_cards, num_opponents=active_opponents)
        obs[59] = _has_flush_draw(hole_cards + board_cards)
        obs[60] = float(max(0.0, min(1.0, equity)))

        hero_seat = bot_player.position % 6
        obs[61 + hero_seat] = 1.0

        active_players = sum(1 for p in game_state.players if p.is_active)
        players_before = sum(1 for p in game_state.players if p.position < bot_player.position and p.is_active)
        players_after = sum(1 for p in game_state.players if p.position > bot_player.position and p.is_active)
        obs[67] = _normalize(float(active_players), 6.0)
        obs[68] = _normalize(float(players_before), 5.0)
        obs[69] = _normalize(float(players_after), 5.0)

        total_pot = float(max(0, game_state.pot))
        current_bet = float(max(0, game_state.current_bet))
        hero_bet = float(max(0, bot_player.bet))
        to_call = max(0.0, current_bet - hero_bet)
        hero_stack = float(max(0, bot_player.stack))
        big_blind = float(max(1, game_state.big_blind))

        active_opp_stacks = [float(max(0, p.stack)) for p in game_state.players if p.is_active and not p.is_bot]
        effective_stack = min([hero_stack] + active_opp_stacks) if active_opp_stacks else hero_stack

        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, bot_player)
        min_raise_to_value = float(min_raise_to) if max_raise_to >= min_raise_to and hero_stack > to_call else 0.0

        obs[70] = _normalize(total_pot / big_blind, 200.0)
        obs[71] = _normalize(to_call / big_blind, 50.0)
        obs[72] = _normalize(min_raise_to_value / big_blind, 200.0)
        obs[73] = _normalize(hero_stack / big_blind, 200.0)
        obs[74] = _normalize(effective_stack / big_blind, 200.0)
        spr = hero_stack / max(total_pot, big_blind)
        obs[75] = _normalize(spr, 20.0)
        obs[76] = float(to_call / max(total_pot + to_call, 1.0))

        hero_contrib = hero_bet
        obs[77] = float(hero_contrib / max(hero_contrib + hero_stack, 1.0))
        obs[78] = 0.0
        obs[79] = _normalize((current_bet / big_blind), 20.0)
        obs[80] = _normalize((hero_contrib / big_blind), 200.0)

        legal_actions = self.get_legal_actions(game_state)
        for action_id in legal_actions:
            if 0 <= action_id < NUM_ACTIONS:
                obs[81 + action_id] = 1.0

        by_seat = {player.position % 6: player for player in game_state.players}
        for offset in range(6):
            seat = (hero_seat + offset) % 6
            seat_player = by_seat.get(seat)
            obs[86 + offset] = 1.0 if seat_player and seat_player.is_active else 0.0

        if street_idx == 0:
            if current_bet <= big_blind:
                obs[92] = 1.0
            elif current_bet <= (3.0 * big_blind):
                obs[93] = 1.0
            else:
                obs[94] = 1.0

        obs[95] = 1.0 if (hero_bet >= current_bet and current_bet > 0 and to_call <= 0.0) else 0.0

        return obs, equity

    def get_legal_actions(self, game_state: GameStateRequest) -> list[int]:
        """
        Determine legal v21 actions from betting state.
        """
        bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
        if bot_index < 0:
            return [ACTION_CHECK]
        return self.get_legal_action_ids_for_actor(game_state, bot_index)

    def calculate_raise_amount(
        self,
        action: int,
        game_state: GameStateRequest,
    ) -> Optional[int]:
        bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
        if bot_index < 0:
            return None
        return self.calculate_raise_amount_for_actor(action, game_state, bot_index)

    def calculate_raise_amount_for_actor(
        self,
        action: int,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> Optional[int]:
        """
        Calculate the raise-to amount for a v21 raise action.
        """
        if action in (ACTION_FOLD, ACTION_CHECK, ACTION_CALL):
            return None

        if actor_index < 0 or actor_index >= len(game_state.players):
            return None
        actor_player = game_state.players[actor_index]

        current_bet = max(0, int(game_state.current_bet))
        hero_bet = max(0, int(actor_player.bet))
        stack = max(0, int(actor_player.stack))
        to_call = max(0, current_bet - hero_bet)
        pot = float(max(0, game_state.pot))

        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, actor_player)
        if stack <= to_call or max_raise_to < min_raise_to:
            return None

        if action == ACTION_RAISE_HALF_POT:
            target = int(round(hero_bet + to_call + (0.5 * pot)))
        elif action == ACTION_RAISE_POT_OR_ALL_IN:
            target = int(round(hero_bet + to_call + pot))
        else:
            return None

        target = max(target, min_raise_to)
        target = min(target, max_raise_to)
        if target <= current_bet:
            if max_raise_to > current_bet:
                target = max_raise_to
            else:
                return None

        return int(target)


# Singleton instance
_game_service: Optional[GameService] = None


def get_game_service() -> GameService:
    """Get the singleton game service instance."""
    global _game_service
    if _game_service is None:
        _game_service = GameService()
    return _game_service
