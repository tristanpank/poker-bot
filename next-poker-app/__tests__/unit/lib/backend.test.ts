import { getBackendBaseUrl } from '../../../app/lib/backend';

// Helper to delete window and restore it
const originalWindow = global.window;

afterEach(() => {
  // Restore original window after each test
  Object.defineProperty(global, 'window', {
    value: originalWindow,
    writable: true,
  });
  delete process.env.NEXT_PUBLIC_BACKEND_URL;
});

describe('getBackendBaseUrl', () => {
  describe('when NEXT_PUBLIC_BACKEND_URL env var is set', () => {
    it('returns the explicit URL', () => {
      process.env.NEXT_PUBLIC_BACKEND_URL = 'https://my-backend.example.com';
      expect(getBackendBaseUrl()).toBe('https://my-backend.example.com');
    });

    it('strips a trailing slash from the explicit URL', () => {
      process.env.NEXT_PUBLIC_BACKEND_URL = 'https://my-backend.example.com/';
      expect(getBackendBaseUrl()).toBe('https://my-backend.example.com');
    });

    it('handles whitespace around the URL', () => {
      process.env.NEXT_PUBLIC_BACKEND_URL = '  https://my-backend.example.com  ';
      expect(getBackendBaseUrl()).toBe('https://my-backend.example.com');
    });
  });

  describe('when NEXT_PUBLIC_BACKEND_URL is not set', () => {
    beforeEach(() => {
      delete process.env.NEXT_PUBLIC_BACKEND_URL;
    });

    it('returns localhost:8000 when hostname is localhost', () => {
      Object.defineProperty(global, 'window', {
        value: { location: { hostname: 'localhost', protocol: 'http:' } },
        writable: true,
      });
      expect(getBackendBaseUrl()).toBe('http://localhost:8000');
    });

    it('returns localhost:8000 when hostname is 127.0.0.1', () => {
      Object.defineProperty(global, 'window', {
        value: { location: { hostname: '127.0.0.1', protocol: 'http:' } },
        writable: true,
      });
      expect(getBackendBaseUrl()).toBe('http://localhost:8000');
    });

    it('derives port-8000 URL from a non-local hostname', () => {
      Object.defineProperty(global, 'window', {
        value: { location: { hostname: '192.168.1.42', protocol: 'http:' } },
        writable: true,
      });
      expect(getBackendBaseUrl()).toBe('http://192.168.1.42:8000');
    });

    it('uses the current protocol for non-local hostnames', () => {
      Object.defineProperty(global, 'window', {
        value: { location: { hostname: 'poker.example.com', protocol: 'https:' } },
        writable: true,
      });
      expect(getBackendBaseUrl()).toBe('https://poker.example.com:8000');
    });

    it('falls back to http://localhost:8000 when window is undefined (SSR)', () => {
      Object.defineProperty(global, 'window', {
        value: undefined,
        writable: true,
      });
      expect(getBackendBaseUrl()).toBe('http://localhost:8000');
    });
  });
});
