import {
  getTablePosition,
  getSeatLabel,
  getDefaultPlayerName,
  compactSeatMap,
  getCompactPositionForSeat,
  getCompactRoleForSeat,
  FULL_RING_SEAT_COUNT,
  tablePositionByPlayers,
} from '../../../app/lib/tablePositions';

describe('tablePositionByPlayers', () => {
  it('defines positions for 2-6 players', () => {
    expect(Object.keys(tablePositionByPlayers).map(Number).sort((a, b) => a - b)).toEqual([2, 3, 4, 5, 6]);
  });
});

describe('FULL_RING_SEAT_COUNT', () => {
  it('is 6', () => {
    expect(FULL_RING_SEAT_COUNT).toBe(6);
  });
});

describe('getTablePosition', () => {
  it('returns correct role for a 6-player table', () => {
    expect(getTablePosition(0, 6)).toBe('SB');
    expect(getTablePosition(1, 6)).toBe('BB');
    expect(getTablePosition(2, 6)).toBe('UTG');
    expect(getTablePosition(3, 6)).toBe('HJ');
    expect(getTablePosition(4, 6)).toBe('CO');
    expect(getTablePosition(5, 6)).toBe('BTN');
  });

  it('returns correct role for a 2-player table', () => {
    expect(getTablePosition(0, 2)).toBe('SB/BTN');
    expect(getTablePosition(1, 2)).toBe('BB');
  });

  it('returns correct role for a 3-player table', () => {
    expect(getTablePosition(0, 3)).toBe('SB');
    expect(getTablePosition(1, 3)).toBe('BB');
    expect(getTablePosition(2, 3)).toBe('BTN');
  });

  it('falls back to 6-player roles for unknown table size', () => {
    // tableSize=7 is not defined; should fall back to 6-player layout
    expect(getTablePosition(0, 7)).toBe('SB');
    expect(getTablePosition(5, 7)).toBe('BTN');
  });

  it('falls back to "P<n+1>" for out-of-bounds position', () => {
    expect(getTablePosition(99, 6)).toBe('P100');
  });
});

describe('getSeatLabel', () => {
  it('returns 1-indexed seat label', () => {
    expect(getSeatLabel(0)).toBe('Seat 1');
    expect(getSeatLabel(1)).toBe('Seat 2');
    expect(getSeatLabel(5)).toBe('Seat 6');
  });
});

describe('getDefaultPlayerName', () => {
  it('returns 1-indexed player name', () => {
    expect(getDefaultPlayerName(0)).toBe('Player 1');
    expect(getDefaultPlayerName(3)).toBe('Player 4');
  });
});

describe('compactSeatMap', () => {
  it('returns sorted unique valid seats', () => {
    expect(compactSeatMap([3, 1, 0, 3])).toEqual([0, 1, 3]);
  });

  it('filters out negative seats', () => {
    expect(compactSeatMap([-1, 0, 1])).toEqual([0, 1]);
  });

  it('filters out seats >= FULL_RING_SEAT_COUNT', () => {
    expect(compactSeatMap([0, 5, 6, 10])).toEqual([0, 5]);
  });

  it('filters non-integer values', () => {
    expect(compactSeatMap([1.5, 2, 3])).toEqual([2, 3]);
  });

  it('returns empty array for empty input', () => {
    expect(compactSeatMap([])).toEqual([]);
  });
});

describe('getCompactPositionForSeat', () => {
  it('returns index of seat in compacted seat map', () => {
    expect(getCompactPositionForSeat(2, [0, 2, 4])).toBe(1);
    expect(getCompactPositionForSeat(0, [0, 2, 4])).toBe(0);
    expect(getCompactPositionForSeat(4, [0, 2, 4])).toBe(2);
  });

  it('returns null when seat is not in the occupied list', () => {
    expect(getCompactPositionForSeat(3, [0, 2, 4])).toBeNull();
  });

  it('returns null for empty occupied seats', () => {
    expect(getCompactPositionForSeat(0, [])).toBeNull();
  });
});

describe('getCompactRoleForSeat', () => {
  it('returns null when seat is not occupied', () => {
    expect(getCompactRoleForSeat(3, [0, 1, 2])).toBeNull();
  });

  it('returns null when fewer than 2 players', () => {
    expect(getCompactRoleForSeat(0, [0])).toBeNull();
  });

  it('returns correct role string for 2-player game', () => {
    // Seats 0 and 1 occupied — position 0 = SB/BTN, position 1 = BB
    expect(getCompactRoleForSeat(0, [0, 1])).toBe('SB/BTN');
    expect(getCompactRoleForSeat(1, [0, 1])).toBe('BB');
  });

  it('returns correct role string for 6-player game', () => {
    const occupied = [0, 1, 2, 3, 4, 5];
    expect(getCompactRoleForSeat(0, occupied)).toBe('SB');
    expect(getCompactRoleForSeat(5, occupied)).toBe('BTN');
  });
});
