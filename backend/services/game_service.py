"""
Game service for managing poker game state and building model observations.

Converts frontend game state to the observation format expected by the trained models.
V19+: Uses range-weighted Monte Carlo equity for more realistic post-flop estimates.
"""

import math
import random
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
from pokerkit import Card, Deck, StandardHighHand

from backend.config import get_settings
from backend.models.schemas import (
    GameStateRequest,
    CardSchema,
    PlayerState,
)
from backend.poker_versions import (
    ACTION_AGGRO_LARGE,
    ACTION_AGGRO_SMALL,
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_RAISE_HALF_POT,
    ACTION_RAISE_POT_OR_ALL_IN,
    V24_FACING_BET_RAISE_TO_MULTIPLIERS,
    V24_NON_ALL_IN_RAISE_ACTIONS,
    V24_POSTFLOP_BET_POT_MULTIPLIERS,
    V24_PREFLOP_OPEN_RAISE_TO_BB,
    get_action_names,
    get_version_spec,
    version_to_int,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_FEATURES_PATH = _REPO_ROOT / "training" / "src" / "features"
if str(_TRAINING_FEATURES_PATH) not in sys.path:
    sys.path.insert(0, str(_TRAINING_FEATURES_PATH))
_TRAINING_WORKERS_PATH = _REPO_ROOT / "training" / "src" / "workers"
if str(_TRAINING_WORKERS_PATH) not in sys.path:
    sys.path.insert(0, str(_TRAINING_WORKERS_PATH))

from preflop_blueprint_v24 import (  # noqa: E402
    BUILTIN_PREFLOP_BLUEPRINT_NAME,
    preflop_blueprint_policy as preflop_blueprint_policy_v24,
)
from position_abstraction import canonical_late_position_index  # noqa: E402
from preflop_blueprint_v25 import preflop_blueprint_policy as preflop_blueprint_policy_v25  # noqa: E402
from poker_state_v24 import encode_info_state as encode_info_state_v24  # noqa: E402
from poker_state_v25 import (  # noqa: E402
    abstract_raise_target as abstract_raise_target_v25,
    build_legal_action_mask as build_legal_action_mask_v25,
    encode_info_state as encode_info_state_v25,
)
from poker_worker_v25 import (  # noqa: E402
    _policy_action_for_snapshot as runtime_policy_action_for_snapshot_v25,
    build_runtime_policy_config as build_runtime_policy_config_v25,
)


FRONTEND_ACTION_FOLD = "fold"
FRONTEND_ACTION_CHECK = "check"
FRONTEND_ACTION_CALL = "call"
FRONTEND_ACTION_RAISE = "raise_amt"

FEATURE_RANKS = "23456789TJQKA"
FEATURE_SUITS = "cdhs"


class _RuntimePot:
    def __init__(self, amount: float):
        self.amount = float(max(0.0, amount))


class _WebsiteRuntimeState:
    def __init__(
        self,
        *,
        actor_index: int,
        board_cards: list[Card],
        hole_cards: list[list[Card]],
        stacks: list[float],
        bets: list[float],
        pots: list[_RuntimePot],
        min_raise_to: int,
        max_raise_to: int,
    ):
        self.actor_index = int(actor_index)
        self.board_cards = list(board_cards)
        self.hole_cards = [list(cards) for cards in hole_cards]
        self.stacks = [float(value) for value in stacks]
        self.bets = [float(value) for value in bets]
        self.pots = list(pots)
        self.min_completion_betting_or_raising_to_amount = int(max(0, min_raise_to))
        self.max_completion_betting_or_raising_to_amount = int(max(0, max_raise_to))
        self.min_completion_betting_or_raising_to = self.min_completion_betting_or_raising_to_amount
        self.max_completion_betting_or_raising_to = self.max_completion_betting_or_raising_to_amount

    def can_fold(self) -> bool:
        if self.actor_index < 0 or self.actor_index >= len(self.bets):
            return False
        return float(max(self.bets) - self.bets[self.actor_index]) > 1e-6

    def can_check_or_call(self) -> bool:
        if self.actor_index < 0 or self.actor_index >= len(self.bets):
            return False
        to_call = float(max(self.bets) - self.bets[self.actor_index])
        if to_call <= 1e-6:
            return True
        return float(self.stacks[self.actor_index]) > 1e-6

    def can_complete_bet_or_raise_to(self) -> bool:
        if self.actor_index < 0 or self.actor_index >= len(self.bets):
            return False
        to_call = max(0.0, float(max(self.bets) - self.bets[self.actor_index]))
        return (
            float(self.stacks[self.actor_index]) > to_call + 1e-6
            and self.max_completion_betting_or_raising_to_amount >= self.min_completion_betting_or_raising_to_amount
            and self.max_completion_betting_or_raising_to_amount > int(max(self.bets))
        )


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


def _street_action_order(player_count: int, street_idx: int) -> list[int]:
    count = max(0, int(player_count))
    if count <= 0:
        return []
    if int(street_idx) == 0:
        start = 0 if count <= 2 else 2
    else:
        start = 1 if count == 2 else 0
    return [int((start + offset) % count) for offset in range(count)]


def _action_order_position_counts(active_flags: list[bool], hero_seat: int, street_idx: int) -> tuple[int, int]:
    order = _street_action_order(len(active_flags), street_idx)
    if hero_seat not in order:
        return 0, 0
    hero_idx = order.index(int(hero_seat))
    players_before = sum(1 for seat in order[:hero_idx] if active_flags[seat])
    players_after = sum(1 for seat in order[hero_idx + 1 :] if active_flags[seat])
    return int(players_before), int(players_after)


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


def _sorted_suit_hist(cards: list[Card]) -> np.ndarray:
    hist = np.zeros(4, dtype=np.float32)
    if not cards:
        return hist
    suit_counts = np.zeros(4, dtype=np.float32)
    for card in cards:
        suit_counts[_suit_index(card)] += 1.0
    hist[:] = np.sort(suit_counts)[::-1]
    hist /= float(len(cards))
    return hist


def _current_hand_strength_scalar(
    hole_cards: list[Card],
    board_cards: list[Card],
    num_opponents: int = 1,
) -> float:
    if len(hole_cards) != 2:
        return 0.0
    if len(board_cards) < 3:
        return 0.0

    cards = hole_cards + board_cards
    rank_counts = [0] * 13
    suit_counts = [0] * 4
    unique_ranks: set[int] = set()
    for card in cards:
        rank_idx = _rank_index(card)
        suit_idx = _suit_index(card)
        rank_counts[rank_idx] += 1
        suit_counts[suit_idx] += 1
        unique_ranks.add(rank_idx)

    max_rank = max(rank_counts)
    pair_count = 0
    has_trips = False
    has_quads = False
    for count in rank_counts:
        if count >= 2:
            pair_count += 1
        if count >= 3:
            has_trips = True
        if count >= 4:
            has_quads = True
    has_flush = max(suit_counts) >= 5

    if 12 in unique_ranks:
        unique_ranks.add(-1)
    ordered = sorted(unique_ranks)
    has_straight = False
    for idx in range(max(0, len(ordered) - 4)):
        window = ordered[idx : idx + 5]
        if len(window) == 5 and window[-1] - window[0] == 4:
            has_straight = True
            break

    if has_flush and has_straight:
        return 0.995
    if has_quads:
        return 0.96
    if has_trips and pair_count >= 2:
        return 0.90
    if has_flush:
        return 0.82
    if has_straight:
        return 0.74
    if has_trips:
        return 0.64
    if pair_count >= 2:
        return 0.50
    if max_rank >= 2:
        return 0.34
    return 0.18


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

    def _resolve_version(self, version: Optional[str], game_state: Optional[GameStateRequest] = None) -> str:
        if version:
            return str(version).lower()
        if game_state is not None and game_state.model_version:
            return str(game_state.model_version).lower()
        return str(self.settings.model_version).lower()

    def _seat_for_player_index(self, game_state: GameStateRequest, player_index: int) -> int:
        if player_index < 0 or player_index >= len(game_state.players):
            return -1
        player_count = max(1, len(game_state.players))
        return int(game_state.players[player_index].position) % player_count

    def _starting_stacks_by_seat(self, game_state: GameStateRequest) -> list[int]:
        player_count = len(game_state.players)
        fallback = [max(0, int(player.stack + player.bet)) for player in game_state.players]
        starting = list(game_state.starting_stacks or [])
        by_seat = [0] * player_count
        if len(starting) != player_count:
            for idx, player in enumerate(game_state.players):
                seat = int(player.position) % player_count
                by_seat[seat] = int(fallback[idx])
            return by_seat
        for idx, player in enumerate(game_state.players):
            seat = int(player.position) % player_count
            by_seat[seat] = max(0, int(starting[idx]))
        return by_seat

    def _street_raise_count(self, game_state: GameStateRequest) -> int:
        explicit = getattr(game_state, "street_raise_count", None)
        if explicit is not None:
            return max(0, int(explicit))
        current_bet = max(0, int(game_state.current_bet))
        big_blind = max(1, int(game_state.big_blind))
        board_len = len(game_state.community_cards)
        if board_len == 0:
            if current_bet <= big_blind:
                return 0
            if current_bet <= (3 * big_blind):
                return 1
            return 2
        if current_bet <= 0:
            return 0
        return 1

    def _estimated_preflop_raise_count(self, game_state: GameStateRequest) -> int:
        explicit = getattr(game_state, "preflop_raise_count", None)
        if explicit is not None:
            return max(0, int(explicit))
        current_bet = max(0, int(game_state.current_bet))
        big_blind = max(1, int(game_state.big_blind))
        if current_bet <= big_blind:
            return 0
        prior_levels = {
            max(0, int(player.bet))
            for player in game_state.players
            if big_blind <= max(0, int(player.bet)) < current_bet
        }
        return int(len(prior_levels))

    def _estimated_preflop_call_count(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> int:
        explicit = getattr(game_state, "preflop_call_count", None)
        if explicit is not None:
            return max(0, int(explicit))
        big_blind = max(1, int(game_state.big_blind))
        if self._estimated_preflop_raise_count(game_state) != 0:
            return 0
        matching_big_blinds = sum(
            1
            for idx, player in enumerate(game_state.players)
            if idx != actor_index and max(0, int(player.bet)) == big_blind
        )
        return int(
            max(
                0,
                matching_big_blinds - 1,
            )
        )

    def _estimated_preflop_last_raiser_position(
        self,
        game_state: GameStateRequest,
    ) -> Optional[int]:
        explicit = getattr(game_state, "preflop_last_raiser", None)
        if explicit is not None:
            return int(explicit)
        if len(game_state.community_cards) != 0:
            return None
        current_bet = max(0, int(game_state.current_bet))
        big_blind = max(1, int(game_state.big_blind))
        if current_bet <= big_blind:
            return None
        player_count = max(1, len(game_state.players))
        by_seat = {int(player.position) % player_count: player for player in game_state.players}
        candidates = [
            seat
            for seat, player in by_seat.items()
            if player.is_active and max(0, int(player.bet)) == current_bet
        ]
        if not candidates:
            return None
        preflop_order = [2, 3, 4, 5, 0, 1]
        indexed = {seat: idx for idx, seat in enumerate(preflop_order)}
        return int(max(candidates, key=lambda seat: indexed.get(int(seat), -1)))

    def _last_aggressor(self, game_state: GameStateRequest) -> Optional[int]:
        explicit = getattr(game_state, "last_aggressor", None)
        if explicit is not None:
            return int(explicit)
        return self._estimated_preflop_last_raiser_position(game_state)

    def _build_v24_runtime_state(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> tuple[int, _WebsiteRuntimeState, SimpleNamespace]:
        if actor_index < 0 or actor_index >= len(game_state.players):
            raise ValueError("Invalid actor index for v24 runtime state")

        player_count = len(game_state.players)
        actor_seat = self._seat_for_player_index(game_state, actor_index)
        board_cards = [card_schema_to_pokerkit(card) for card in game_state.community_cards]
        hole_cards_by_seat: list[list[Card]] = [[] for _ in range(player_count)]
        stacks_by_seat = [0.0] * player_count
        bets_by_seat = [0.0] * player_count
        in_hand = [False] * player_count
        starting_stacks = self._starting_stacks_by_seat(game_state)

        for idx, player in enumerate(game_state.players):
            seat = int(player.position) % player_count
            stacks_by_seat[seat] = float(max(0, int(player.stack)))
            bets_by_seat[seat] = float(max(0, int(player.bet)))
            in_hand[seat] = bool(player.is_active)
            if player.hole_cards:
                hole_cards_by_seat[seat] = [card_schema_to_pokerkit(card) for card in player.hole_cards]

        actor_player = game_state.players[actor_index]
        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, actor_player)
        pot_without_live_bets = max(0.0, float(game_state.pot) - float(sum(bets_by_seat)))
        state = _WebsiteRuntimeState(
            actor_index=actor_seat,
            board_cards=board_cards,
            hole_cards=hole_cards_by_seat,
            stacks=stacks_by_seat,
            bets=bets_by_seat,
            pots=[_RuntimePot(pot_without_live_bets)],
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
        )
        contributions = [
            max(0.0, float(starting_stacks[seat]) - float(stacks_by_seat[seat]))
            for seat in range(player_count)
        ]
        hand_ctx = SimpleNamespace(
            starting_stacks=list(starting_stacks),
            big_blind=int(game_state.big_blind),
            small_blind=max(1, int(game_state.big_blind) // 2),
            in_hand=list(in_hand),
            contributions=contributions,
            hole_cards_by_seat=hole_cards_by_seat,
            current_street=get_street_from_board(board_cards),
            street_raise_count=int(self._street_raise_count(game_state)),
            preflop_raise_count=int(self._estimated_preflop_raise_count(game_state)),
            preflop_call_count=int(self._estimated_preflop_call_count(game_state, actor_index)),
            preflop_last_raiser=self._estimated_preflop_last_raiser_position(game_state),
            last_aggressor=self._last_aggressor(game_state),
        )
        return actor_seat, state, hand_ctx

    def _all_in_allowed_for_actor(self, game_state: GameStateRequest, actor_index: int) -> bool:
        if actor_index < 0 or actor_index >= len(game_state.players):
            return False
        if len(game_state.community_cards) != 0:
            return True
        actor = game_state.players[actor_index]
        big_blind = max(1.0, float(game_state.big_blind))
        hero_stack = float(max(0, int(actor.stack)))
        opponent_stacks = [
            float(max(0, int(player.stack)))
            for idx, player in enumerate(game_state.players)
            if idx != actor_index and player.is_active
        ]
        effective_stack = min([hero_stack] + opponent_stacks) if opponent_stacks else hero_stack
        effective_stack_bb = effective_stack / big_blind
        if effective_stack_bb <= 25.0:
            return True
        return self._estimated_preflop_raise_count(game_state) >= 2

    def _v24_raise_context(self, game_state: GameStateRequest, actor_index: int) -> str:
        actor_player = game_state.players[actor_index]
        street = get_street_from_board(game_state.community_cards)
        current_bet = max(0, int(game_state.current_bet))
        actor_bet = max(0, int(actor_player.bet))
        to_call = max(0, current_bet - actor_bet)
        if street == 0:
            if self._estimated_preflop_raise_count(game_state) == 0:
                return "preflop_open"
            return "preflop_raise"
        if to_call > 0:
            return "postflop_raise"
        return "postflop_bet"

    def _v24_raise_target(
        self,
        action: int,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> Optional[int]:
        if actor_index < 0 or actor_index >= len(game_state.players):
            return None
        actor_player = game_state.players[actor_index]
        current_bet = max(0, int(game_state.current_bet))
        actor_bet = max(0, int(actor_player.bet))
        stack = max(0, int(actor_player.stack))
        to_call = max(0, current_bet - actor_bet)
        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, actor_player)
        if stack <= to_call or max_raise_to < min_raise_to:
            return None
        if action not in V24_NON_ALL_IN_RAISE_ACTIONS:
            return None

        pot = float(max(0, game_state.pot))
        big_blind = float(max(1, int(game_state.big_blind)))
        context = self._v24_raise_context(game_state, actor_index)
        if context == "preflop_open":
            base_size_bb = V24_PREFLOP_OPEN_RAISE_TO_BB.get(int(action))
            if base_size_bb is None:
                return None
            limper_bonus_bb = min(3.0, float(self._estimated_preflop_call_count(game_state, actor_index)))
            target = int(round((base_size_bb + limper_bonus_bb) * big_blind))
        elif context == "postflop_bet":
            multiplier = V24_POSTFLOP_BET_POT_MULTIPLIERS.get(int(action))
            if multiplier is None:
                return None
            target = int(round(actor_bet + to_call + (multiplier * pot)))
        else:
            multiplier = V24_FACING_BET_RAISE_TO_MULTIPLIERS.get(int(action))
            if multiplier is None:
                return None
            target = int(round(current_bet * multiplier))
        target = max(int(min_raise_to), min(target, int(max_raise_to)))
        return int(target)

    def _v24_legal_raise_actions(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> list[int]:
        if actor_index < 0 or actor_index >= len(game_state.players):
            return []
        actor_player = game_state.players[actor_index]
        current_bet = max(0, int(game_state.current_bet))
        actor_bet = max(0, int(actor_player.bet))
        stack = max(0, int(actor_player.stack))
        to_call = max(0, current_bet - actor_bet)
        min_raise_to, max_raise_to = self._get_raise_bounds(game_state, actor_player)
        if stack <= to_call or max_raise_to < min_raise_to or max_raise_to <= current_bet:
            return []

        actions: list[int] = []
        seen_targets: set[int] = set()
        for action_id in V24_NON_ALL_IN_RAISE_ACTIONS:
            target = self._v24_raise_target(action_id, game_state, actor_index)
            if target is None:
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            actions.append(action_id)
        return actions

    def _v25_raise_target(
        self,
        action: int,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> Optional[int]:
        try:
            actor_seat, runtime_state, hand_ctx = self._build_v24_runtime_state(game_state, actor_index)
        except ValueError:
            return None
        if runtime_state.actor_index != actor_seat:
            return None
        target = abstract_raise_target_v25(runtime_state, int(action), hand_ctx)
        return None if target is None else int(target)

    def _v25_legal_raise_actions(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> list[int]:
        try:
            actor_seat, runtime_state, hand_ctx = self._build_v24_runtime_state(game_state, actor_index)
        except ValueError:
            return []
        legal_mask = build_legal_action_mask_v25(runtime_state, actor_seat, hand_ctx)
        return [
            int(action_id)
            for action_id in np.flatnonzero(np.asarray(legal_mask, dtype=np.float32) > 0.5)
            if int(action_id) >= 3
        ]

    def _effective_stack_bb_for_actor(
        self,
        game_state: GameStateRequest,
        actor_index: int,
    ) -> float:
        actor = game_state.players[actor_index]
        big_blind = max(1.0, float(game_state.big_blind))
        hero_stack = float(max(0, int(actor.stack)))
        opponent_stacks = [
            float(max(0, int(player.stack)))
            for idx, player in enumerate(game_state.players)
            if idx != actor_index and player.is_active
        ]
        effective_stack = min([hero_stack] + opponent_stacks) if opponent_stacks else hero_stack
        return float(effective_stack / big_blind)

    def get_preflop_blueprint_recommendation(
        self,
        game_state: GameStateRequest,
        actor_index: int,
        version: Optional[str] = None,
    ) -> Optional[dict[str, object]]:
        resolved_version = self._resolve_version(version, game_state)
        if version_to_int(resolved_version) < 24:
            return None
        if len(game_state.community_cards) != 0:
            return None
        if actor_index < 0 or actor_index >= len(game_state.players):
            return None

        actor = game_state.players[actor_index]
        if actor.hole_cards is None or len(actor.hole_cards) != 2:
            return None

        legal_actions = self.get_legal_action_ids_for_actor(game_state, actor_index, version=resolved_version)
        if not legal_actions:
            return None

        spec = get_version_spec(resolved_version)
        legal_mask = np.zeros(max(spec.action_dim, max(legal_actions) + 1), dtype=np.float32)
        for action_id in legal_actions:
            legal_mask[int(action_id)] = 1.0

        hole_cards = [card_schema_to_pokerkit(card) for card in actor.hole_cards]
        to_call_bb = float(max(0, int(game_state.current_bet) - int(actor.bet))) / float(max(1, int(game_state.big_blind)))
        blueprint_policy = preflop_blueprint_policy_v25 if version_to_int(resolved_version) >= 25 else preflop_blueprint_policy_v24
        player_count = max(2, len(game_state.players))
        policy, meta = blueprint_policy(
            hole_cards=hole_cards,
            actor_seat=int(actor.position) % player_count,
            legal_mask=legal_mask,
            effective_stack_bb=self._effective_stack_bb_for_actor(game_state, actor_index),
            to_call_bb=to_call_bb,
            preflop_raise_count=int(self._estimated_preflop_raise_count(game_state)),
            preflop_call_count=int(self._estimated_preflop_call_count(game_state, actor_index)),
            aggressor_seat=self._estimated_preflop_last_raiser_position(game_state),
            player_count=player_count,
            blueprint_name=BUILTIN_PREFLOP_BLUEPRINT_NAME,
        )
        if float(np.asarray(policy, dtype=np.float32).sum()) <= 1e-8 or not bool(meta.get("covered", False)):
            return None
        action_id = int(np.argmax(policy))
        return {
            "action_id": action_id,
            "policy": np.asarray(policy, dtype=np.float32),
            "meta": meta,
        }
    
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

        if current_bet <= 0:
            min_raise_to = max(big_blind, hero_bet + big_blind)
            return int(min_raise_to), int(max_raise_to)

        prior_highest_below = 0
        for player in game_state.players:
            player_bet = max(0, int(player.bet))
            if player_bet < current_bet:
                prior_highest_below = max(prior_highest_below, player_bet)
        if current_bet <= big_blind:
            min_increment = big_blind
        else:
            min_increment = max(big_blind, current_bet - prior_highest_below)
        min_raise_to = current_bet + min_increment

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
        version: Optional[str] = None,
    ) -> list[int]:
        resolved_version = self._resolve_version(version, game_state)
        version_num = version_to_int(resolved_version)
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
                if version_num >= 25:
                    action_ids.extend(self._v25_legal_raise_actions(game_state, actor_index))
                elif version_num >= 24:
                    action_ids.extend(self._v24_legal_raise_actions(game_state, actor_index))
                else:
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
        actor_seat = self._seat_for_player_index(next_state, actor_index)
        street = get_street_from_board(next_state.community_cards)
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
            if street == 0 and to_call > 0:
                next_state.preflop_call_count += 1
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
            next_state.last_aggressor = actor_seat
            next_state.street_raise_count += 1
            if street == 0:
                next_state.preflop_raise_count += 1
                next_state.preflop_last_raiser = actor_seat

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
        current_seat_map = list(game_state.seat_map or [])

        next_players: list[PlayerState] = []
        next_seat_map: list[int] = []
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
            if 0 <= old_index < len(current_seat_map):
                next_seat_map.append(int(current_seat_map[old_index]))
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
            seat_map=next_seat_map or None,
            starting_stacks=[int(player.stack) for player in next_players],
            current_bet=0,
            big_blind=int(game_state.big_blind),
            current_player_idx=next_actor_idx,
            street_raise_count=0,
            preflop_raise_count=0,
            preflop_call_count=0,
            preflop_last_raiser=None,
            last_aggressor=None,
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

    def build_observation(
        self,
        game_state: GameStateRequest,
        version: Optional[str] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Build the version-appropriate Deep CFR observation vector and equity estimate.
        """
        resolved_version = self._resolve_version(version, game_state)
        spec = get_version_spec(resolved_version)

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

        if version_to_int(resolved_version) >= 25:
            bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
            if bot_index < 0:
                raise ValueError("No bot player found in game state")
            bot_seat, runtime_state, hand_ctx = self._build_v24_runtime_state(game_state, bot_index)
            return encode_info_state_v25(runtime_state, bot_seat, hand_ctx), equity

        if version_to_int(resolved_version) >= 24:
            bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
            if bot_index < 0:
                raise ValueError("No bot player found in game state")
            bot_seat, runtime_state, hand_ctx = self._build_v24_runtime_state(game_state, bot_index)
            return encode_info_state_v24(runtime_state, bot_seat, hand_ctx), equity

        obs = np.zeros(spec.state_dim, dtype=np.float32)

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
            obs[45:49] = _sorted_suit_hist(board_cards)

        obs[53] = 1.0 if np.any(board_rank_counts >= 2) else 0.0
        obs[54] = 1.0 if np.any(board_rank_counts >= 3) else 0.0
        obs[55] = 1.0 if board_len >= 3 and np.any(board_suit_counts == board_len) else 0.0
        obs[56] = 1.0 if board_len >= 3 and np.count_nonzero(board_suit_counts) == 2 else 0.0
        obs[57] = _board_connected(board_cards)

        active_opponents = max(1, sum(1 for p in game_state.players if p.is_active and not p.is_bot))
        obs[58] = _estimate_preflop_strength(hole_cards, num_opponents=active_opponents)
        obs[59] = _has_flush_draw(hole_cards + board_cards)
        obs[60] = _current_hand_strength_scalar(hole_cards, board_cards, num_opponents=active_opponents)

        player_count = max(1, len(game_state.players))
        hero_seat = bot_player.position % player_count
        hero_position_bucket = canonical_late_position_index(player_count, hero_seat)
        obs[61 + hero_position_bucket] = 1.0

        active_flags = [False] * player_count
        for player in game_state.players:
            seat = int(player.position) % player_count
            active_flags[seat] = bool(player.is_active)
        active_players = sum(1 for flag in active_flags if flag)
        players_before, players_after = _action_order_position_counts(active_flags, hero_seat, street_idx)
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
        street_raise_count = self._street_raise_count(game_state)
        obs[77] = float(hero_contrib / max(hero_contrib + hero_stack, 1.0))
        obs[78] = _normalize(float(street_raise_count), 4.0)
        obs[79] = _normalize((current_bet / big_blind), 20.0)
        obs[80] = _normalize((hero_contrib / big_blind), 200.0)

        legal_actions = self.get_legal_actions(game_state, version=resolved_version)
        if spec.summarized_legal_features:
            obs[81] = 1.0 if ACTION_FOLD in legal_actions else 0.0
            obs[82] = 1.0 if ACTION_CHECK in legal_actions else 0.0
            obs[83] = 1.0 if ACTION_CALL in legal_actions else 0.0
            obs[84] = 1.0 if ACTION_AGGRO_SMALL in legal_actions else 0.0
            obs[85] = 1.0 if ACTION_AGGRO_LARGE in legal_actions else 0.0
        else:
            for action_id in legal_actions:
                if 0 <= action_id < min(spec.action_dim, 5):
                    obs[81 + action_id] = 1.0

        by_seat = {player.position % max(1, player_count): player for player in game_state.players}
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

    def get_runtime_action_for_actor(
        self,
        game_state: GameStateRequest,
        actor_index: int,
        version: Optional[str] = None,
    ) -> tuple[int, dict[str, float]]:
        resolved_version = self._resolve_version(version, game_state)
        if version_to_int(resolved_version) < 25:
            raise ValueError("Runtime subgame resolving is only available for v25+")
        if actor_index < 0 or actor_index >= len(game_state.players):
            raise ValueError("Invalid actor index")

        try:
            from backend.services.model_service import get_model_service
        except Exception as exc:
            raise RuntimeError(f"Model service unavailable: {exc}") from exc

        model_service = get_model_service()
        snapshot = model_service.load_model(resolved_version)
        actor_seat, runtime_state, hand_ctx = self._build_v24_runtime_state(game_state, actor_index)
        runtime_config = build_runtime_policy_config_v25(
            {
                "evaluation_mode": "self_play",
                "eval_hero_seat": actor_seat,
                "runtime_subgame_resolving_enabled": True,
                "runtime_subgame_resolving_hero_only": True,
                "runtime_subgame_traversals": 32,
                "runtime_subgame_use_average_policy": True,
            }
        )
        action_id, details = runtime_policy_action_for_snapshot_v25(
            snapshot,
            runtime_state,
            actor_seat,
            hand_ctx,
            random.Random(0),
            config=runtime_config,
            return_details=True,
            sample_action=False,
        )
        policy = np.asarray(details.get("policy", ()), dtype=np.float32).reshape(-1)
        action_names = get_action_names(resolved_version)
        score_dict = {
            action_names.get(idx, f"ACTION_{idx}"): float(policy[idx]) if idx < len(policy) else 0.0
            for idx in range(len(action_names))
        }
        return int(action_id), score_dict

    def get_legal_actions(self, game_state: GameStateRequest, version: Optional[str] = None) -> list[int]:
        """
        Determine legal v21 actions from betting state.
        """
        bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
        if bot_index < 0:
            return [ACTION_CHECK]
        return self.get_legal_action_ids_for_actor(game_state, bot_index, version=version)

    def calculate_raise_amount(
        self,
        action: int,
        game_state: GameStateRequest,
        version: Optional[str] = None,
    ) -> Optional[int]:
        bot_index = next((idx for idx, player in enumerate(game_state.players) if player.is_bot), -1)
        if bot_index < 0:
            return None
        return self.calculate_raise_amount_for_actor(action, game_state, bot_index, version=version)

    def calculate_raise_amount_for_actor(
        self,
        action: int,
        game_state: GameStateRequest,
        actor_index: int,
        version: Optional[str] = None,
    ) -> Optional[int]:
        """
        Calculate the raise-to amount for the selected abstract action.
        """
        resolved_version = self._resolve_version(version, game_state)
        version_num = version_to_int(resolved_version)
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

        if version_num >= 25:
            target = self._v25_raise_target(action, game_state, actor_index)
            if target is None:
                return None
        elif version_num >= 24:
            target = self._v24_raise_target(action, game_state, actor_index)
            if target is None:
                return None
        elif action == ACTION_RAISE_HALF_POT:
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
