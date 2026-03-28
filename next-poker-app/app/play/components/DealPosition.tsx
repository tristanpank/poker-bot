import React from 'react';
import { getTablePosition } from '../../lib/tablePositions';

type DealPositionProps = {
    tableSize: number;
    onSelectSeat: (seatIndex: number) => void;
    canUndo: boolean;
    onUndo: () => void;
};

export default function DealPosition({ tableSize, onSelectSeat, canUndo, onUndo }: DealPositionProps) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-3xl font-bold tracking-tight text-[var(--color-text-primary)] mb-2">Bot Position</h1>
                <p className="text-[var(--color-text-secondary)] text-xs tracking-[0.2em] uppercase font-semibold">Select where the bot is sitting</p>
            </div>
            <div className="w-full max-w-sm grid grid-cols-3 gap-3 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                {Array.from({ length: tableSize }, (_, i) => (
                    <button key={i} onClick={() => onSelectSeat(i)}
                        className="p-5 rounded-2xl border-2 transition-all duration-200 flex flex-col items-center gap-1 bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-accent)]/15 hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] active:scale-95">
                        <span className="text-2xl font-bold">{getTablePosition(i, tableSize)}</span>
                        <span className="text-[9px] uppercase tracking-wider text-[var(--color-text-secondary)]">Position {i + 1}</span>
                    </button>
                ))}
            </div>
            {canUndo && (
                <button onClick={onUndo} className="mt-4 px-4 py-2 rounded-xl text-sm font-semibold bg-slate-800 text-slate-300 border border-slate-600 hover:bg-slate-700 active:scale-95 transition-all">
                    ↩ Undo
                </button>
            )}
        </div>
    );
}
