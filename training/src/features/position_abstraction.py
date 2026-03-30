from __future__ import annotations


# Preserve the blind seats and collapse the remaining occupied seats toward
# later 6-max positions so short-handed tables do not look artificially early.
#
# Examples:
# - 3-max: SB, BB, BTN
# - 4-max: SB, BB, CO, BTN
# - 5-max: SB, BB, MP, CO, BTN
#
# Heads-up keeps the raw seat indices so the small blind/button can still use
# blind-specific preflop logic.
_CANONICAL_LATE_POSITIONS_BY_TABLE_SIZE: dict[int, tuple[int, ...]] = {
    2: (0, 1),
    3: (0, 1, 5),
    4: (0, 1, 4, 5),
    5: (0, 1, 3, 4, 5),
    6: (0, 1, 2, 3, 4, 5),
}


def canonical_late_position_index(player_count: int, seat_index: int) -> int:
    count = max(2, min(6, int(player_count)))
    mapping = _CANONICAL_LATE_POSITIONS_BY_TABLE_SIZE[count]
    seat = int(seat_index)
    if seat < 0:
        return mapping[0]
    if seat >= len(mapping):
        return mapping[-1]
    return int(mapping[seat])
