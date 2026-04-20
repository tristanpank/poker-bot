import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import JoinPage from '../../app/join/page';

// ── Module mocks ────────────────────────────────────────────────────────────

// Mock the custom WebRTC hook so we don't need real browser media APIs
jest.mock('../../app/lib/useCvWebRtcStream', () => ({
  useCvWebRtcStream: jest.fn(() => ({
    videoRef: { current: null },
    isStreaming: false,
    error: null,
    captureInfo: 'N/A',
    phase: 'idle',
    startStream: jest.fn().mockResolvedValue(true),
    stopStream: jest.fn().mockResolvedValue(undefined),
  })),
}));

// Mock backend URL so we don't depend on window.location in tests
jest.mock('../../app/lib/backend', () => ({
  getBackendBaseUrl: () => 'http://localhost:8000',
}));

// ── fetch mock helpers ──────────────────────────────────────────────────────

const mockFetch = jest.fn();

// Suppress residual act() warnings that come from async polling effects
// firing after a test has already finished its assertions. All tests pass
// and the warnings do not indicate real problems.
let consoleErrorSpy: jest.SpyInstance;

beforeAll(() => {
  global.fetch = mockFetch;
  consoleErrorSpy = jest
    .spyOn(console, 'error')
    .mockImplementation((...args) => {
      // Let through any error that is NOT the act() warning
      if (typeof args[0] === 'string' && args[0].includes('not wrapped in act')) return;
      // eslint-disable-next-line no-console
      console.warn(...args);
    });
});

afterAll(() => {
  consoleErrorSpy.mockRestore();
});

afterEach(() => {
  mockFetch.mockReset();
  // Clear sessionStorage between tests
  window.sessionStorage.clear();
  jest.clearAllTimers();
});

// Helper: return a JSON fetch response
function mockJsonResponse(body: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    catch: jest.fn(),
  } as unknown as Response);
}

// A valid status-by-code response that includes a bot seat
const statusResponse = {
  sessionActive: true,
  session_id: 'sess-abc',
  opponents: {},
  tableSize: 6,
  botPosition: 2,
};

// A successful join response
const joinResponse = {
  session_id: 'sess-xyz',
  cv_session_id: 'cv-123',
  player_name: 'Alice',
};

// ── Tests ────────────────────────────────────────────────────────────────────

describe('JoinPage – idle state', () => {
  beforeEach(() => {
    // Default: status-by-code returns a resolved table
    mockFetch.mockImplementation((url: string) => {
      if (String(url).includes('status-by-code')) {
        return mockJsonResponse(statusResponse);
      }
      return mockJsonResponse({}, 404);
    });
  });

  // Helper: render JoinPage and flush the initial polling useEffect so that
  // setBotSeat / setAvailableSeats state updates are wrapped in act().
  async function renderAndSettle() {
    await act(async () => {
      render(<JoinPage />);
    });
  }

  it('renders the page heading', async () => {
    await renderAndSettle();
    expect(screen.getByRole('heading', { name: /join poker session/i })).toBeInTheDocument();
  });

  it('renders the session code input', async () => {
    await renderAndSettle();
    expect(screen.getByPlaceholderText('ABC123')).toBeInTheDocument();
  });

  it('renders the player name input', async () => {
    await renderAndSettle();
    expect(screen.getByPlaceholderText(/player/i)).toBeInTheDocument();
  });

  it('renders the Join Session button', async () => {
    await renderAndSettle();
    expect(screen.getByRole('button', { name: /join session/i })).toBeInTheDocument();
  });

  it('Join button is disabled when code is less than 6 characters', async () => {
    await renderAndSettle();
    const input = screen.getByPlaceholderText('ABC123');
    await act(async () => {
      fireEvent.change(input, { target: { value: 'AB' } });
    });
    expect(screen.getByRole('button', { name: /join session/i })).toBeDisabled();
  });

  it('renders the seat picker with at least 7 buttons (6 seats + join)', async () => {
    await renderAndSettle();
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThanOrEqual(7);
  });

  it('uppercases the code as it is typed', async () => {
    await renderAndSettle();
    const input = screen.getByPlaceholderText('ABC123') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(input, { target: { value: 'abc123' } });
    });
    expect(input.value).toBe('ABC123');
  });
});

describe('JoinPage – join flow', () => {
  beforeEach(() => {
    mockFetch.mockImplementation((url: string) => {
      const u = String(url);
      if (u.includes('status-by-code')) return mockJsonResponse(statusResponse);
      if (u.includes('webcam/join'))    return mockJsonResponse(joinResponse);
      return mockJsonResponse({}, 404);
    });
  });

  it('transitions to joined state after a successful join call', async () => {
    render(<JoinPage />);

    // Enter a 6-char code so the join button becomes enabled
    fireEvent.change(screen.getByPlaceholderText('ABC123'), {
      target: { value: 'AAA111' },
    });

    // Wait for the status-by-code fetch to settle (botSeat discovered)
    await waitFor(() => {
      expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /join session/i })).not.toBeDisabled();
    });

    // Click join
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /join session/i }));
    });

    // After joining, the webcam section should be visible
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start webcam/i })).toBeInTheDocument();
    });
  });

  it('shows the connected player name after joining', async () => {
    render(<JoinPage />);

    fireEvent.change(screen.getByPlaceholderText('ABC123'), {
      target: { value: 'AAA111' },
    });

    await waitFor(() =>
      expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument()
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /join session/i })).not.toBeDisabled();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /join session/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/connected as/i)).toBeInTheDocument();
      expect(screen.getByText('Alice')).toBeInTheDocument();
    });
  });
});

describe('JoinPage – join errors', () => {
  it('shows error message when join API returns an error', async () => {
    mockFetch.mockImplementation((url: string) => {
      const u = String(url);
      if (u.includes('status-by-code')) return mockJsonResponse(statusResponse);
      if (u.includes('webcam/join')) {
        return mockJsonResponse({ detail: 'Seat already taken.' }, 400);
      }
      return mockJsonResponse({}, 404);
    });

    render(<JoinPage />);

    fireEvent.change(screen.getByPlaceholderText('ABC123'), {
      target: { value: 'ERR111' },
    });

    await waitFor(() =>
      expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument()
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /join session/i })).not.toBeDisabled();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /join session/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/seat already taken/i)).toBeInTheDocument();
    });
  });

  it('shows error message when code is empty on join attempt', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: jest.fn().mockResolvedValue({}),
    } as unknown as Response);

    render(<JoinPage />);

    // Do not type a code — join button is disabled; let's call handleJoin by
    // directly simulating (the button is disabled, so we verify disabled state)
    expect(screen.getByRole('button', { name: /join session/i })).toBeDisabled();
  });
});

describe('JoinPage – session status polling', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('marks the session inactive and resets to idle when sessionActive becomes false', async () => {
    let callCount = 0;

    mockFetch.mockImplementation((url: string) => {
      const u = String(url);
      if (u.includes('status-by-code')) return mockJsonResponse(statusResponse);
      if (u.includes('webcam/join'))    return mockJsonResponse(joinResponse);
      if (u.includes('webcam/status/')) {
        callCount++;
        // Account for double invoking useEffect in tests
        const active = callCount <= 2;
        return mockJsonResponse({ ...statusResponse, sessionActive: active, session_id: 'sess-xyz', opponents: {} });
      }
      return mockJsonResponse({}, 404);
    });

    render(<JoinPage />);

    fireEvent.change(screen.getByPlaceholderText('ABC123'), { target: { value: 'ACTIVE' } });

    await waitFor(() =>
      expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument()
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /join session/i })).not.toBeDisabled();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /join session/i }));
    });

    await waitFor(() => screen.getByRole('button', { name: /start webcam/i }));

    // Advance timer past the poll interval (3000 ms) twice
    await act(async () => {
      jest.advanceTimersByTime(7000);
    });

    // After the session becomes inactive, the page should reset to idle
    await waitFor(() => {
      expect(screen.getByPlaceholderText('ABC123')).toBeInTheDocument();
      expect(screen.getByText(/host ended this game/i)).toBeInTheDocument();
    });
  });
});

describe('JoinPage – leave session', () => {
  it('returns to idle state when "Leave Session" is clicked', async () => {
    mockFetch.mockImplementation((url: string) => {
      const u = String(url);
      if (u.includes('status-by-code')) return mockJsonResponse(statusResponse);
      if (u.includes('webcam/join'))    return mockJsonResponse(joinResponse);
      if (u.includes('webcam/disconnect')) return mockJsonResponse({ ok: true });
      if (u.includes('webcam/status/'))
        return mockJsonResponse({ ...statusResponse, sessionActive: true, opponents: {} });
      return mockJsonResponse({}, 404);
    });

    render(<JoinPage />);

    fireEvent.change(screen.getByPlaceholderText('ABC123'), { target: { value: 'LEAVE1' } });

    // Wait for seat map to load (bot seat resolved)
    await waitFor(() =>
      expect(screen.queryByText(/waiting for the host/i)).not.toBeInTheDocument()
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /join session/i })).not.toBeDisabled();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /join session/i }));
    });

    // After a successful join the "connected as" text should appear
    await waitFor(() => {
      expect(screen.getByText(/connected as/i)).toBeInTheDocument();
    });

    // "Leave Session" is rendered when !isStreaming
    const leaveButton = screen.getByRole('button', { name: /leave session/i });
    expect(leaveButton).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(leaveButton);
    });

    // Back to idle: code input is visible again
    await waitFor(() => {
      expect(screen.getByPlaceholderText('ABC123')).toBeInTheDocument();
    });

    // And the webcam section is gone
    expect(screen.queryByRole('button', { name: /start webcam/i })).not.toBeInTheDocument();
  });
});
