const nextJest = require('next/jest');

const createJestConfig = nextJest({
  // Loads next.config.js and .env files for the test environment
  dir: './',
});

/** @type {import('jest').Config} */
const config = {
  coverageProvider: 'v8',
  testEnvironment: 'jsdom',
  // Run jest.setup.ts after the test framework is installed in each suite
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  testMatch: ['<rootDir>/__tests__/**/*.(test|spec).(ts|tsx)'],
};

module.exports = createJestConfig(config);
