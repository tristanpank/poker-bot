from __future__ import annotations

ACTION_NAMES_V26 = ("fold", "check_call", "raise")
ACTION_COUNT_V26 = len(ACTION_NAMES_V26)
POSITION_NAMES_V26 = ("SB", "BB", "UTG", "MP", "CO", "BTN")

__all__ = [
    "ACTION_COUNT_V26",
    "ACTION_NAMES_V26",
    "POSITION_NAMES_V26",
]
