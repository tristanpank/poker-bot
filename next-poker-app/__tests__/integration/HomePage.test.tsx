import React from 'react';
import { render, screen } from '@testing-library/react';
import HomePage from '../../app/page';

// next/link renders an <a> tag; no mocking needed in jsdom
describe('HomePage', () => {
  it('renders the "Welcome" heading', () => {
    render(<HomePage />);
    expect(screen.getByRole('heading', { name: /welcome/i })).toBeInTheDocument();
  });

  it('renders "Poker Bot AI" badge label', () => {
    render(<HomePage />);
    expect(screen.getByText(/poker bot ai/i)).toBeInTheDocument();
  });

  it('renders a "Create Game" link', () => {
    render(<HomePage />);
    const link = screen.getByRole('link', { name: /create game/i });
    expect(link).toBeInTheDocument();
  });

  it('"Create Game" link points to /play', () => {
    render(<HomePage />);
    const link = screen.getByRole('link', { name: /create game/i });
    expect(link).toHaveAttribute('href', '/play');
  });

  it('renders a "Join Game" link', () => {
    render(<HomePage />);
    const link = screen.getByRole('link', { name: /join game/i });
    expect(link).toBeInTheDocument();
  });

  it('"Join Game" link points to /join', () => {
    render(<HomePage />);
    const link = screen.getByRole('link', { name: /join game/i });
    expect(link).toHaveAttribute('href', '/join');
  });

  it('renders a description about starting or joining a session', () => {
    render(<HomePage />);
    expect(screen.getByText(/create a new bot operating session/i)).toBeInTheDocument();
  });
});
