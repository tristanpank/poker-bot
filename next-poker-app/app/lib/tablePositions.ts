export const tablePositionByPlayers: Record<number, string[]> = {
  2: ['SB/BTN', 'BB'],
  3: ['SB', 'BB', 'BTN'],
  4: ['SB', 'BB', 'UTG', 'BTN'],
  5: ['SB', 'BB', 'UTG', 'CO', 'BTN'],
  6: ['SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'],
};

export const FULL_RING_SEAT_COUNT = 6;

export const sixSeatLayout = [
  { seat: 0, className: 'left-1/2 top-2 -translate-x-1/2' },
  { seat: 1, className: 'right-4 top-16' },
  { seat: 2, className: 'right-6 bottom-16' },
  { seat: 3, className: 'left-1/2 bottom-2 -translate-x-1/2' },
  { seat: 4, className: 'left-6 bottom-16' },
  { seat: 5, className: 'left-4 top-16' },
] as const;

export function getTablePosition(position: number, tableSize: number): string {
  const roles = tablePositionByPlayers[tableSize] ?? tablePositionByPlayers[6];
  return roles[position] ?? `P${position + 1}`;
}

export function getSeatLabel(position: number): string {
  return `Seat ${position + 1}`;
}

export function getDefaultPlayerName(position: number): string {
  return `Player ${position + 1}`;
}

export function compactSeatMap(seats: number[]): number[] {
  return Array.from(
    new Set(
      seats.filter((seat) => Number.isInteger(seat) && seat >= 0 && seat < FULL_RING_SEAT_COUNT),
    ),
  ).sort((a, b) => a - b);
}

export function getCompactPositionForSeat(seat: number, occupiedSeats: number[]): number | null {
  const seatMap = compactSeatMap(occupiedSeats);
  const position = seatMap.indexOf(seat);
  return position >= 0 ? position : null;
}

export function getCompactRoleForSeat(seat: number, occupiedSeats: number[]): string | null {
  const position = getCompactPositionForSeat(seat, occupiedSeats);
  if (position === null) {
    return null;
  }
  const playerCount = compactSeatMap(occupiedSeats).length;
  if (playerCount < 2) {
    return null;
  }
  return getTablePosition(position, playerCount);
}
