"""
Shared poker model version metadata for backend/runtime integration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_RAISE_HALF_POT = 3
ACTION_RAISE_POT_OR_ALL_IN = 4

ACTION_RAISE_QUARTER_POT = 3
ACTION_RAISE_THREE_QUARTER_POT = 5
ACTION_RAISE_POT = 6
ACTION_RAISE_125_POT = 7
ACTION_RAISE_150_POT = 8
ACTION_RAISE_200_POT = 9
ACTION_ALL_IN = 10

ACTION_NAMES_V21 = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE_HALF_POT",
    4: "RAISE_POT_OR_ALL_IN",
}

ACTION_NAMES_V24 = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE_QUARTER_POT",
    4: "RAISE_HALF_POT",
    5: "RAISE_THREE_QUARTER_POT",
    6: "RAISE_POT",
    7: "RAISE_125_POT",
    8: "RAISE_150_POT",
    9: "RAISE_200_POT",
    10: "ALL_IN",
}

ACTION_NAMES = ACTION_NAMES_V24

V24_NON_ALL_IN_RAISE_ACTIONS = (
    ACTION_RAISE_QUARTER_POT,
    ACTION_RAISE_HALF_POT,
    ACTION_RAISE_THREE_QUARTER_POT,
    ACTION_RAISE_POT,
    ACTION_RAISE_125_POT,
    ACTION_RAISE_150_POT,
    ACTION_RAISE_200_POT,
)

V24_RAISE_MULTIPLIERS = {
    ACTION_RAISE_QUARTER_POT: 0.25,
    ACTION_RAISE_HALF_POT: 0.50,
    ACTION_RAISE_THREE_QUARTER_POT: 0.75,
    ACTION_RAISE_POT: 1.00,
    ACTION_RAISE_125_POT: 1.25,
    ACTION_RAISE_150_POT: 1.50,
    ACTION_RAISE_200_POT: 2.00,
}


@dataclass(frozen=True)
class PokerVersionSpec:
    version_floor: int
    state_dim: int
    action_dim: int
    action_names: dict[int, str]
    opponent_profile_dim: int = 0
    summarized_legal_features: bool = False


DEEP_CFR_V21_SPEC = PokerVersionSpec(
    version_floor=21,
    state_dim=98,
    action_dim=5,
    action_names=ACTION_NAMES_V21,
)

DEEP_CFR_V23_SPEC = PokerVersionSpec(
    version_floor=23,
    state_dim=153,
    action_dim=5,
    action_names=ACTION_NAMES_V21,
    opponent_profile_dim=55,
)

DEEP_CFR_V24_SPEC = PokerVersionSpec(
    version_floor=24,
    state_dim=153,
    action_dim=11,
    action_names=ACTION_NAMES_V24,
    opponent_profile_dim=55,
    summarized_legal_features=True,
)

LEGACY_V15_SPEC = PokerVersionSpec(
    version_floor=15,
    state_dim=520,
    action_dim=6,
    action_names={idx: f"ACTION_{idx}" for idx in range(6)},
)

LEGACY_V13_SPEC = PokerVersionSpec(
    version_floor=13,
    state_dim=385,
    action_dim=6,
    action_names={idx: f"ACTION_{idx}" for idx in range(6)},
)


def version_to_int(version: str | None) -> int:
    match = re.search(r"(\d+)", (version or "").lower())
    if match is None:
        return 0
    return int(match.group(1))


def get_version_spec(version: str | None) -> PokerVersionSpec:
    version_num = version_to_int(version)
    if version_num >= 24:
        return DEEP_CFR_V24_SPEC
    if version_num >= 23:
        return DEEP_CFR_V23_SPEC
    if version_num >= 21:
        return DEEP_CFR_V21_SPEC
    if version_num >= 15:
        return LEGACY_V15_SPEC
    return LEGACY_V13_SPEC


def get_action_names(version: str | None) -> dict[int, str]:
    return get_version_spec(version).action_names
