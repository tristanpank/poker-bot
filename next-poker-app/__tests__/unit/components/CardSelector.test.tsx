import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import CardSelector from '../../../app/play/components/CardSelector';

// Default props for a minimal valid render
const baseProps = {
  pickingFor: 'hole' as const,
  holeCardsCount: 0,
  communityCardsCount: 0,
  usedCards: new Set<string>(),
  pendingRank: null,
  setPendingRank: jest.fn(),
  onSelectCard: jest.fn(),
  onCancel: jest.fn(),
};

describe('CardSelector', () => {
  describe('when pickingFor is null', () => {
    it('renders nothing', () => {
      const { container } = render(
        <CardSelector {...baseProps} pickingFor={null} />
      );
      expect(container).toBeEmptyDOMElement();
    });
  });

  describe('rank picker (no pendingRank)', () => {
    it('shows all 13 rank buttons', () => {
      render(<CardSelector {...baseProps} />);
      const ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
      ranks.forEach((rank) => {
        expect(screen.getByRole('button', { name: rank })).toBeInTheDocument();
      });
    });

    it('shows correct label for hole card 1 of 2', () => {
      render(<CardSelector {...baseProps} holeCardsCount={0} />);
      expect(screen.getByText(/select hole card 1 of 2/i)).toBeInTheDocument();
    });

    it('shows correct label for hole card 2 of 2', () => {
      render(<CardSelector {...baseProps} holeCardsCount={1} />);
      expect(screen.getByText(/select hole card 2 of 2/i)).toBeInTheDocument();
    });

    it('shows correct label for community flop card', () => {
      render(<CardSelector {...baseProps} pickingFor="community" communityCardsCount={0} />);
      expect(screen.getByText(/1\/3 flop/i)).toBeInTheDocument();
    });

    it('shows correct label for community turn card', () => {
      render(<CardSelector {...baseProps} pickingFor="community" communityCardsCount={3} />);
      expect(screen.getByText(/turn/i)).toBeInTheDocument();
    });

    it('shows correct label for community river card', () => {
      render(<CardSelector {...baseProps} pickingFor="community" communityCardsCount={4} />);
      expect(screen.getByText(/river/i)).toBeInTheDocument();
    });

    it('shows correct label for showdown mode', () => {
      render(<CardSelector {...baseProps} pickingFor="showdown" holeCardsCount={0} />);
      expect(screen.getByText(/select showdown card 1 of 2/i)).toBeInTheDocument();
    });

    it('disables a rank button when all 4 suits are used', () => {
      const usedCards = new Set(['As', 'Ah', 'Ad', 'Ac']);
      render(<CardSelector {...baseProps} usedCards={usedCards} />);
      const aceButton = screen.getByRole('button', { name: 'A' });
      expect(aceButton).toBeDisabled();
    });

    it('does not disable a rank button when only some suits are used', () => {
      const usedCards = new Set(['As', 'Ah']);
      render(<CardSelector {...baseProps} usedCards={usedCards} />);
      const aceButton = screen.getByRole('button', { name: 'A' });
      expect(aceButton).not.toBeDisabled();
    });

    it('calls setPendingRank with the clicked rank', () => {
      const setPendingRank = jest.fn();
      render(<CardSelector {...baseProps} setPendingRank={setPendingRank} />);
      fireEvent.click(screen.getByRole('button', { name: 'K' }));
      expect(setPendingRank).toHaveBeenCalledWith('K');
    });

    it('calls onCancel when Cancel is clicked', () => {
      const onCancel = jest.fn();
      render(<CardSelector {...baseProps} onCancel={onCancel} />);
      fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
      expect(onCancel).toHaveBeenCalledTimes(1);
    });
  });

  describe('suit picker (with pendingRank)', () => {
    const propsWithPendingRank = { ...baseProps, pendingRank: 'A' };

    it('shows suit picker with all 4 suits', () => {
      render(<CardSelector {...propsWithPendingRank} />);
      expect(screen.getByRole('button', { name: /spades/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /hearts/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /diamonds/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /clubs/i })).toBeInTheDocument();
    });

    it('shows the pending rank in the heading', () => {
      render(<CardSelector {...propsWithPendingRank} />);
      expect(screen.getByText(/pick suit for A/i)).toBeInTheDocument();
    });

    it('disables a suit button when that card is already used', () => {
      const usedCards = new Set(['As']);
      render(<CardSelector {...propsWithPendingRank} usedCards={usedCards} />);
      expect(screen.getByRole('button', { name: /spades/i })).toBeDisabled();
    });

    it('calls onSelectCard with rank and suit when a suit is clicked', () => {
      const onSelectCard = jest.fn();
      render(<CardSelector {...propsWithPendingRank} onSelectCard={onSelectCard} />);
      fireEvent.click(screen.getByRole('button', { name: /hearts/i }));
      expect(onSelectCard).toHaveBeenCalledWith('A', 'h');
    });

    it('calls setPendingRank(null) when "Back to ranks" is clicked', () => {
      const setPendingRank = jest.fn();
      render(<CardSelector {...propsWithPendingRank} setPendingRank={setPendingRank} />);
      fireEvent.click(screen.getByRole('button', { name: /back to ranks/i }));
      expect(setPendingRank).toHaveBeenCalledWith(null);
    });
  });
});
