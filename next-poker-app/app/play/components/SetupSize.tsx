import React from 'react';

type SetupSizeProps = {
    tableSize: number;
    setTableSize: (size: number) => void;
    onContinue: () => void;
};

export default function SetupSize({ tableSize, setTableSize, onContinue }: SetupSizeProps) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-10 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-3 flex items-center justify-center gap-3">
                    <span className="text-5xl text-[var(--color-accent)]">♠</span>
                    PokerBot
                </h1>
                <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.2em] uppercase font-semibold">Select Table Size</p>
            </div>
            <div className="w-full max-w-sm grid grid-cols-2 gap-4 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
                {[2, 3, 4, 5, 6].map(size => (
                    <button key={size} onClick={() => setTableSize(size)}
                        className={`p-6 rounded-2xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-1 ${tableSize === size
                            ? 'bg-[var(--color-accent)]/15 border-[var(--color-accent)] shadow-[0_0_20px_var(--color-accent-glow)] text-[var(--color-accent)] scale-105'
                            : 'bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)] hover:border-[var(--color-text-secondary)]'}`}>
                        <span className="text-4xl font-bold">{size}</span>
                        <span className={`text-[10px] uppercase tracking-wider font-semibold ${tableSize === size ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}>Max Players</span>
                    </button>
                ))}
            </div>
            <div className="w-full max-w-sm mt-6 animate-slide-up" style={{ animationDelay: '300ms', animationFillMode: 'forwards' }}>
                <button onClick={onContinue}
                    className="w-full py-5 px-6 rounded-2xl font-bold text-lg tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-3">
                    Continue <span className="text-xl">→</span>
                </button>
            </div>
        </div>
    );
}
