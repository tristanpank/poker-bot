import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PlayPhase from '../../../app/play/components/PlayPhase';
import { TableSeatVisual } from '../../../app/play/components/TableVisual';

// ── Minimal prop factory ────────────────────────────────────────────────────

type Card = { rank: string; suit: string };

const makePlayer = (overrides: Partial<{
  position: number; stack: number; bet: number;
  hole_cards: Card[] | null; is_bot: boolean;
  is_active: boolean; has_acted: boolean;
}> = {}) => ({
  position: 0,
  stack: 1000,
  bet: 0,
  hole_cards: null,
  is_bot: false,
  is_active: true,
  has_acted: false,
  ...overrides,
});

const makeSeat = (seat: number): TableSeatVisual => ({
  seat,
  title: `Seat ${seat + 1}`,
  subtitle: 'Player',
  tone: 'normal',
  onClick: null,
});

const defaultLegalActions = {
  canFold: true,
  canCheck: false,
  canCall: true,
  canRaise: true,
  toCall: 20,
  minRaiseTo: 40,
  maxRaiseTo: 1000,
};

const defaultProps = {
  pot: 100,
  currentBet: 20,
  bigBlind: 10,
  botPosition: 0,
  holeCards: [{ rank: 'A', suit: 's' }, { rank: 'K', suit: 'h' }] as Card[],
  communityCards: [] as Card[],
  street: 'preflop' as const,
  players: [makePlayer({ is_bot: false }), makePlayer({ position: 1, is_bot: true })],
  seatMap: [0, 1],
  playerNames: { '0': 'Alice' },
  tableSeats: [makeSeat(0), makeSeat(1)],
  currentPlayerIdx: 0,       // Human player's turn by default
  isLoading: false,
  botResponse: null,
  showRaiseInput: false,
  setShowRaiseInput: jest.fn(),
  raiseInput: '',
  setRaiseInput: jest.fn(),
  onQueryBot: jest.fn(),
  onRecordAction: jest.fn(),
  onUndo: jest.fn(),
  canUndo: false,
  legalActions: defaultLegalActions,
  showdownMode: false,
  showdownEntries: [],
  currentShowdownPlayerIndex: null,
  showdownCanResolve: false,
  isResolvingShowdown: false,
  showdownError: null,
  showdownResult: null,
  resultFlash: null,
  onMuckShowdown: jest.fn(),
  onClearShowdown: jest.fn(),
  onResolveShowdown: jest.fn(),
};

// ── Header ──────────────────────────────────────────────────────────────────

describe('PlayPhase – header', () => {
  it('shows "PokerBot" brand name', () => {
    render(<PlayPhase {...defaultProps} />);
    expect(screen.getByText('PokerBot')).toBeInTheDocument();
  });

  it('displays the current pot value', () => {
    render(<PlayPhase {...defaultProps} pot={250} />);
    expect(screen.getByText('250')).toBeInTheDocument();
  });

  it('displays the current bet', () => {
    render(<PlayPhase {...defaultProps} currentBet={40} />);
    expect(screen.getByText('40')).toBeInTheDocument();
  });

  it('displays the big blind', () => {
    render(<PlayPhase {...defaultProps} bigBlind={5} />);
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('shows bot stack when a bot player exists', () => {
    const players = [
      makePlayer({ is_bot: false }),
      makePlayer({ position: 1, is_bot: true, stack: 750 }),
    ];
    render(<PlayPhase {...defaultProps} players={players} />);
    expect(screen.getByText('750')).toBeInTheDocument();
  });
});

// ── Turn display ─────────────────────────────────────────────────────────────

describe('PlayPhase – turn display', () => {
  it('shows "Bot Turn" when it is the bot\'s turn', () => {
    const players = [
      makePlayer({ is_bot: false }),
      makePlayer({ position: 1, is_bot: true }),
    ];
    render(<PlayPhase {...defaultProps} players={players} currentPlayerIdx={1} />);
    expect(screen.getByText(/bot turn/i)).toBeInTheDocument();
  });

  it('shows the human player name when it is a human\'s turn', () => {
    render(<PlayPhase {...defaultProps} currentPlayerIdx={0} />);
    // getPlayerName builds "Alice (SB)" for position 0 in a 2-player game
    expect(screen.getByText(/alice/i)).toBeInTheDocument();
  });

  it('shows "Betting Round Complete" when currentPlayerIdx is -1', () => {
    render(<PlayPhase {...defaultProps} currentPlayerIdx={-1} />);
    expect(screen.getByText(/betting round complete/i)).toBeInTheDocument();
  });
});

// ── Human action buttons ─────────────────────────────────────────────────────

describe('PlayPhase – human action buttons', () => {
  it('renders Fold, Call/Check, and Raise buttons', () => {
    render(<PlayPhase {...defaultProps} />);
    expect(screen.getByRole('button', { name: /fold/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /call/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /raise/i })).toBeInTheDocument();
  });

  it('shows "Call" when canCall is true', () => {
    render(<PlayPhase {...defaultProps} legalActions={{ ...defaultLegalActions, canCall: true, canCheck: false }} />);
    expect(screen.getByRole('button', { name: /^call$/i })).toBeInTheDocument();
  });

  it('shows "Check" when canCheck is true and canCall is false', () => {
    render(
      <PlayPhase
        {...defaultProps}
        legalActions={{ ...defaultLegalActions, canCall: false, canCheck: true }}
      />
    );
    expect(screen.getByRole('button', { name: /^check$/i })).toBeInTheDocument();
  });

  it('disables Fold when canFold is false', () => {
    render(<PlayPhase {...defaultProps} legalActions={{ ...defaultLegalActions, canFold: false }} />);
    expect(screen.getByRole('button', { name: /fold/i })).toBeDisabled();
  });

  it('disables Raise when canRaise is false', () => {
    render(<PlayPhase {...defaultProps} legalActions={{ ...defaultLegalActions, canRaise: false }} />);
    expect(screen.getByRole('button', { name: /raise/i })).toBeDisabled();
  });

  it('calls onRecordAction("fold") when Fold is clicked', () => {
    const onRecordAction = jest.fn();
    render(<PlayPhase {...defaultProps} onRecordAction={onRecordAction} />);
    fireEvent.click(screen.getByRole('button', { name: /fold/i }));
    expect(onRecordAction).toHaveBeenCalledWith('fold');
  });

  it('calls onRecordAction("check_call") when Call is clicked', () => {
    const onRecordAction = jest.fn();
    render(<PlayPhase {...defaultProps} onRecordAction={onRecordAction} />);
    fireEvent.click(screen.getByRole('button', { name: /call/i }));
    expect(onRecordAction).toHaveBeenCalledWith('check_call');
  });

  it('calls setShowRaiseInput when Raise is clicked', () => {
    const setShowRaiseInput = jest.fn();
    render(<PlayPhase {...defaultProps} setShowRaiseInput={setShowRaiseInput} showRaiseInput={false} />);
    fireEvent.click(screen.getByRole('button', { name: /raise/i }));
    expect(setShowRaiseInput).toHaveBeenCalledWith(true);
  });
});

// ── Raise input ───────────────────────────────────────────────────────────────

describe('PlayPhase – raise input', () => {
  it('shows raise input when showRaiseInput is true and canRaise is true', () => {
    render(
      <PlayPhase
        {...defaultProps}
        showRaiseInput
        raiseInput="50"
        legalActions={{ ...defaultLegalActions, canRaise: true }}
      />
    );
    expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /confirm/i })).toBeInTheDocument();
  });

  it('does not show raise input when showRaiseInput is false', () => {
    render(<PlayPhase {...defaultProps} showRaiseInput={false} />);
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
  });

  it('disables Confirm when raise value is below minimum', () => {
    render(
      <PlayPhase
        {...defaultProps}
        showRaiseInput
        raiseInput="5"   // below minRaiseTo=40
        legalActions={{ ...defaultLegalActions, canRaise: true, minRaiseTo: 40, maxRaiseTo: 1000 }}
      />
    );
    expect(screen.getByRole('button', { name: /confirm/i })).toBeDisabled();
  });

  it('enables Confirm when raise value is within valid range', () => {
    render(
      <PlayPhase
        {...defaultProps}
        showRaiseInput
        raiseInput="100"  // 40 <= 100 <= 1000
        legalActions={{ ...defaultLegalActions, canRaise: true, minRaiseTo: 40, maxRaiseTo: 1000 }}
      />
    );
    expect(screen.getByRole('button', { name: /confirm/i })).not.toBeDisabled();
  });

  it('calls onRecordAction("raise", amount) when Confirm is clicked with valid value', () => {
    const onRecordAction = jest.fn();
    render(
      <PlayPhase
        {...defaultProps}
        showRaiseInput
        raiseInput="100"
        onRecordAction={onRecordAction}
        legalActions={{ ...defaultLegalActions, canRaise: true, minRaiseTo: 40, maxRaiseTo: 1000 }}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /confirm/i }));
    expect(onRecordAction).toHaveBeenCalledWith('raise', 100);
  });
});

// ── Bot query button ──────────────────────────────────────────────────────────

describe('PlayPhase – bot turn', () => {
  const botTurnProps = {
    ...defaultProps,
    players: [
      makePlayer({ is_bot: false }),
      makePlayer({ position: 1, is_bot: true }),
    ],
    currentPlayerIdx: 1, // Bot's turn
  };

  it('renders "Get Bot Action" button when it is the bot\'s turn', () => {
    render(<PlayPhase {...botTurnProps} />);
    expect(screen.getByRole('button', { name: /get bot action/i })).toBeInTheDocument();
  });

  it('shows "Thinking..." on the button when isLoading', () => {
    render(<PlayPhase {...botTurnProps} isLoading />);
    expect(screen.getByRole('button', { name: /thinking/i })).toBeInTheDocument();
  });

  it('disables "Get Bot Action" button when isLoading', () => {
    render(<PlayPhase {...botTurnProps} isLoading />);
    expect(screen.getByRole('button', { name: /thinking/i })).toBeDisabled();
  });

  it('calls onQueryBot when the button is clicked', () => {
    const onQueryBot = jest.fn();
    render(<PlayPhase {...botTurnProps} onQueryBot={onQueryBot} />);
    fireEvent.click(screen.getByRole('button', { name: /get bot action/i }));
    expect(onQueryBot).toHaveBeenCalledTimes(1);
  });
});

// ── Bot recommendation panel ──────────────────────────────────────────────────

describe('PlayPhase – bot recommendation', () => {
  const botResponse = {
    action: 'RAISE_33_POT',
    action_id: 2,
    amount: 150,
    originalAction: null,
    originalActionId: null,
    originalAmount: null,
    cvInfluenceApplied: false,
    cvActMax: null,
    cvBluffRiskLevel: null as null,
  };

  it('shows recommendation section when botResponse is set and not loading', () => {
    render(<PlayPhase {...defaultProps} botResponse={botResponse} isLoading={false} />);
    expect(screen.getByText(/agent recommendation/i)).toBeInTheDocument();
  });

  it('displays the action label as "RAISE"', () => {
    render(<PlayPhase {...defaultProps} botResponse={botResponse} isLoading={false} />);
    expect(screen.getByText('RAISE')).toBeInTheDocument();
  });

  it('displays the raise amount', () => {
    render(<PlayPhase {...defaultProps} botResponse={botResponse} isLoading={false} />);
    expect(screen.getByText(/amount: 150/i)).toBeInTheDocument();
  });

  it('shows "No bluff detected." when risk level is null', () => {
    render(<PlayPhase {...defaultProps} botResponse={botResponse} isLoading={false} />);
    expect(screen.getByText(/no bluff detected/i)).toBeInTheDocument();
  });

  it('does not show recommendation panel when isLoading is true', () => {
    render(<PlayPhase {...defaultProps} botResponse={botResponse} isLoading />);
    expect(screen.queryByText(/agent recommendation/i)).not.toBeInTheDocument();
  });

  it('shows loading spinner when isLoading', () => {
    render(<PlayPhase {...defaultProps} isLoading />);
    expect(screen.getByText(/thinking\.\.\./i)).toBeInTheDocument();
  });
});

// ── Undo button ───────────────────────────────────────────────────────────────

describe('PlayPhase – undo', () => {
  it('shows undo button when canUndo is true', () => {
    render(<PlayPhase {...defaultProps} canUndo undoLabel="Fold" />);
    expect(screen.getByRole('button', { name: /undo/i })).toBeInTheDocument();
  });

  it('hides undo button when canUndo is false', () => {
    render(<PlayPhase {...defaultProps} canUndo={false} />);
    expect(screen.queryByRole('button', { name: /undo/i })).not.toBeInTheDocument();
  });

  it('calls onUndo when the button is clicked', () => {
    const onUndo = jest.fn();
    render(<PlayPhase {...defaultProps} canUndo onUndo={onUndo} />);
    fireEvent.click(screen.getByRole('button', { name: /undo/i }));
    expect(onUndo).toHaveBeenCalledTimes(1);
  });
});

// ── Result flash ──────────────────────────────────────────────────────────────

describe('PlayPhase – result flash', () => {
  it('shows positive delta with "+" prefix when result is "won"', () => {
    render(<PlayPhase {...defaultProps} resultFlash={{ result: 'won', delta: 50 }} />);
    expect(screen.getByText('+50')).toBeInTheDocument();
  });

  it('shows negative delta without "+" prefix when result is "lost"', () => {
    render(<PlayPhase {...defaultProps} resultFlash={{ result: 'lost', delta: -30 }} />);
    expect(screen.getByText('-30')).toBeInTheDocument();
  });
});

// ── Showdown mode ─────────────────────────────────────────────────────────────

describe('PlayPhase – showdown mode', () => {
  const showdownProps = {
    ...defaultProps,
    showdownMode: true,
    showdownEntries: [
      { playerIndex: 0, position: 0, seat: 0, cards: [], mucked: false },
    ],
    currentShowdownPlayerIndex: 0,
    showdownCanResolve: true,
  };

  it('renders the showdown section', () => {
    render(<PlayPhase {...showdownProps} />);
    expect(screen.getByText(/opponent reveals/i)).toBeInTheDocument();
  });

  it('renders "Resolve Showdown" button when result is not yet set', () => {
    render(<PlayPhase {...showdownProps} showdownResult={null} />);
    expect(screen.getByRole('button', { name: /resolve showdown/i })).toBeInTheDocument();
  });

  it('disables "Resolve Showdown" when showdownCanResolve is false', () => {
    render(<PlayPhase {...showdownProps} showdownCanResolve={false} />);
    expect(screen.getByRole('button', { name: /resolve showdown/i })).toBeDisabled();
  });

  it('calls onResolveShowdown when the button is clicked', () => {
    const onResolveShowdown = jest.fn();
    render(<PlayPhase {...showdownProps} onResolveShowdown={onResolveShowdown} />);
    fireEvent.click(screen.getByRole('button', { name: /resolve showdown/i }));
    expect(onResolveShowdown).toHaveBeenCalledTimes(1);
  });

  it('shows showdown error message', () => {
    render(<PlayPhase {...showdownProps} showdownError="Failed to resolve" />);
    expect(screen.getByText(/failed to resolve/i)).toBeInTheDocument();
  });

  it('shows showdown result when resolved', () => {
    render(
      <PlayPhase
        {...showdownProps}
        showdownResult={{ result: 'won', amount: 200, delta: 100 }}
      />
    );
    expect(screen.getByText(/WON 200/i)).toBeInTheDocument();
  });
});

// ── Street display ────────────────────────────────────────────────────────────

describe('PlayPhase – street display', () => {
  it('shows the current street name in the table center', () => {
    render(<PlayPhase {...defaultProps} street="flop" />);
    expect(screen.getByText('flop')).toBeInTheDocument();
  });
});
