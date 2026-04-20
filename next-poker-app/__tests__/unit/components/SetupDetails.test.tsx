import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SetupDetails from '../../../app/play/components/SetupDetails';

describe('SetupDetails', () => {
    const defaultProps = {
        hasSession: false,
        sessionStacks: [1000],
        sessionProfit: 0,
        smallBlind: 1,
        setSmallBlind: jest.fn(),
        bigBlind: 2,
        setBigBlind: jest.fn(),
        buyIn: 200,
        setBuyIn: jest.fn(),
        onBack: jest.fn(),
        onStart: jest.fn(),
        onEnd: jest.fn()
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('No session state', () => {
        it('renders initial setup view', () => {
            render(<SetupDetails {...defaultProps} />);
            
            expect(screen.getByText('Game Details')).toBeInTheDocument();
            expect(screen.getByText('Configure Your Session')).toBeInTheDocument();
            
            // Check input fields are present
            expect(screen.getByDisplayValue('1')).toBeInTheDocument();
            expect(screen.getByDisplayValue('2')).toBeInTheDocument();
            expect(screen.getByDisplayValue('200')).toBeInTheDocument();
            
            // Buttons
            expect(screen.getByRole('button', { name: /start session/i })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /back/i })).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: /end/i })).not.toBeInTheDocument();
        });

        it('disables back button if showBack is false', () => {
            render(<SetupDetails {...defaultProps} showBack={false} />);
            expect(screen.queryByRole('button', { name: /back/i })).not.toBeInTheDocument();
        });

        it('calls onStart when start button is clicked', () => {
            render(<SetupDetails {...defaultProps} />);
            fireEvent.click(screen.getByRole('button', { name: /start session/i }));
            expect(defaultProps.onStart).toHaveBeenCalledTimes(1);
        });

        it('calls onBack when back button is clicked', () => {
            render(<SetupDetails {...defaultProps} />);
            fireEvent.click(screen.getByRole('button', { name: /back/i }));
            expect(defaultProps.onBack).toHaveBeenCalledTimes(1);
        });

        it('calls setters when inputs change', () => {
            render(<SetupDetails {...defaultProps} />);
            
            fireEvent.change(screen.getByDisplayValue('1'), { target: { value: '5' } });
            expect(defaultProps.setSmallBlind).toHaveBeenCalledWith(5);

            fireEvent.change(screen.getByDisplayValue('2'), { target: { value: '10' } });
            expect(defaultProps.setBigBlind).toHaveBeenCalledWith(10);

            fireEvent.change(screen.getByDisplayValue('200'), { target: { value: '1000' } });
            expect(defaultProps.setBuyIn).toHaveBeenCalledWith(1000);
        });
    });

    describe('Active session state', () => {
        const sessionProps = {
            ...defaultProps,
            hasSession: true,
            sessionStacks: [1500],
            sessionProfit: 500
        };

        it('renders session summary', () => {
            render(<SetupDetails {...sessionProps} />);
            
            expect(screen.getByText('Between Hands')).toBeInTheDocument();
            expect(screen.getByText('1500')).toBeInTheDocument(); // Bot stack
            expect(screen.getByText('+500 session')).toBeInTheDocument(); // Profit
            
            // Should not show inputs
            expect(screen.queryByDisplayValue('1')).not.toBeInTheDocument();
            
            // Buttons
            expect(screen.getByRole('button', { name: /new hand/i })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /end/i })).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: /back/i })).not.toBeInTheDocument();
        });

        it('displays negative profit correctly', () => {
            render(<SetupDetails {...sessionProps} sessionProfit={-200} />);
            expect(screen.getByText('-200 session')).toBeInTheDocument();
        });

        it('calls onStart for new hand', () => {
            render(<SetupDetails {...sessionProps} />);
            fireEvent.click(screen.getByRole('button', { name: /new hand/i }));
            expect(sessionProps.onStart).toHaveBeenCalledTimes(1);
        });

        it('calls onEnd when end button is clicked', () => {
            render(<SetupDetails {...sessionProps} />);
            fireEvent.click(screen.getByRole('button', { name: /end/i }));
            expect(sessionProps.onEnd).toHaveBeenCalledTimes(1);
        });
    });
});
