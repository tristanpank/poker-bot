export const tablePositionByPlayers: Record<number, string[]> = {
  2: ['SB/BTN', 'BB'],
  3: ['SB', 'BB', 'BTN'],
  4: ['SB', 'BB', 'UTG', 'BTN'],
  5: ['SB', 'BB', 'UTG', 'CO', 'BTN'],
  6: ['SB', 'BB', 'UTG', 'HJ', 'CO', 'BTN'],
};

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
