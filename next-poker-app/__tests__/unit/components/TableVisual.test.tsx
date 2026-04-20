import React from 'react';
import { render, screen } from '@testing-library/react';
import TableVisual, { TableSeatVisual } from '../../../app/play/components/TableVisual';

const makeSeat = (overrides: Partial<TableSeatVisual> & { seat: number }): TableSeatVisual => ({
  seat: overrides.seat,
  title: overrides.title ?? `Seat ${overrides.seat + 1}`,
  subtitle: overrides.subtitle ?? 'Player',
  detail: overrides.detail ?? null,
  tone: overrides.tone ?? 'normal',
  onClick: overrides.onClick ?? null,
  isDealer: overrides.isDealer ?? false,
  isBot: overrides.isBot ?? false,
  disabled: overrides.disabled ?? false,
  canAcceptDealerDrop: overrides.canAcceptDealerDrop ?? false,
  dealerDraggable: overrides.dealerDraggable ?? false,
});

const centerNode = <div>Center Content</div>;

describe('TableVisual', () => {
  it('renders the center node', () => {
    render(<TableVisual seats={[]} center={centerNode} />);
    expect(screen.getByText('Center Content')).toBeInTheDocument();
  });

  it('renders all 6 seat buttons (fallback for missing seats)', () => {
    render(<TableVisual seats={[]} center={centerNode} />);
    // sixSeatLayout defines 6 seats; unmapped ones show "Seat N" fallback
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(6);
  });

  it('shows provided seat title and subtitle', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, title: 'BOT', subtitle: '1000' }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.getByText('BOT')).toBeInTheDocument();
    expect(screen.getByText('1000')).toBeInTheDocument();
  });

  it('shows fallback title "Seat N" for unmapped seats', () => {
    render(<TableVisual seats={[]} center={centerNode} />);
    expect(screen.getByText('Seat 1')).toBeInTheDocument();
  });

  it('shows fallback subtitle "Open" for unmapped seats', () => {
    render(<TableVisual seats={[]} center={centerNode} />);
    expect(screen.getAllByText('Open')).toHaveLength(6);
  });

  it('shows dealer badge "D" when isDealer is true', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, isDealer: true }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.getByTitle('Dealer Button')).toBeInTheDocument();
    expect(screen.getByTitle('Dealer Button')).toHaveTextContent('D');
  });

  it('does not show dealer badge when isDealer is false', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, isDealer: false }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.queryByTitle('Dealer Button')).not.toBeInTheDocument();
  });

  it('shows crown 👑 icon when isBot is true', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, isBot: true }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.getByTitle('Host')).toBeInTheDocument();
  });

  it('shows crown 👑 icon when tone is "bot"', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, tone: 'bot' }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.getByTitle('Host')).toBeInTheDocument();
  });

  it('does not show crown icon when isBot is false and tone is not "bot"', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, tone: 'normal' }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.queryByTitle('Host')).not.toBeInTheDocument();
  });

  it('calls onClick when a clickable seat is clicked', () => {
    const onClick = jest.fn();
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, onClick }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    // The first button in the list corresponds to seat 0 in the layout
    screen.getAllByRole('button').forEach((btn) => {
      if (btn.textContent?.includes('Seat 1')) {
        btn.click();
      }
    });
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('does not call onClick when seat is disabled', () => {
    const onClick = jest.fn();
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, onClick, disabled: true }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    const buttons = screen.getAllByRole('button');
    buttons.forEach((btn) => btn.click());
    expect(onClick).not.toHaveBeenCalled();
  });

  it('renders seat detail text when provided', () => {
    const seats: TableSeatVisual[] = [
      makeSeat({ seat: 0, detail: 'BTN' }),
    ];
    render(<TableVisual seats={seats} center={centerNode} />);
    expect(screen.getByText('BTN')).toBeInTheDocument();
  });
});
