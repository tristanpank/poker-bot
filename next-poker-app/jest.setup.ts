// Extends Jest's expect with jest-dom matchers
import '@testing-library/jest-dom';

// Mock window.matchMedia which is not implemented in jsdom
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),       // deprecated
    removeListener: jest.fn(),    // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Clear all mocks between tests
beforeEach(() => {
  jest.clearAllMocks();
});
