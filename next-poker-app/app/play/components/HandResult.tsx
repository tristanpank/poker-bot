import React from 'react';

type HandResultProps = {
    resultType: 'won' | 'lost' | null;
    setResultType: (type: 'won' | 'lost' | null) => void;
    resultAmt: string;
    setResultAmt: (amt: string) => void;
    onConfirm: (won: boolean, amount: number) => void;
    onUndo: () => void;
};

export default function HandResult({
    resultType, setResultType, resultAmt, setResultAmt, onConfirm, onUndo
}: HandResultProps) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">Hand Result</h1>
                <p className="text-[var(--color-text-secondary)] text-xs">How did the bot do?</p>
            </div>

            <div className="w-full max-w-sm flex gap-3">
                <button onClick={() => setResultType('won')}
                    className={`flex-1 py-5 rounded-2xl font-bold text-lg uppercase transition-all duration-200 active:scale-95
          ${resultType === 'won' ? 'bg-[var(--color-accent)] text-slate-950 shadow-[0_0_20px_var(--color-accent-glow)]'
                            : 'bg-emerald-500/10 border border-emerald-500/30 text-[var(--color-accent)] hover:bg-emerald-500/20'}`}>
                    Won
                </button>
                <button onClick={() => setResultType('lost')}
                    className={`flex-1 py-5 rounded-2xl font-bold text-lg uppercase transition-all duration-200 active:scale-95
          ${resultType === 'lost' ? 'bg-[var(--color-danger)] text-white shadow-[0_0_20px_var(--color-danger-glow)]'
                            : 'bg-red-500/10 border border-red-500/30 text-[var(--color-danger)] hover:bg-red-500/20'}`}>
                    Lost
                </button>
            </div>

            {resultType && (
                <div className="w-full max-w-sm flex flex-col gap-4 animate-fade-in">
                    <div className="flex items-center gap-3">
                        <label className="text-sm text-[var(--color-text-secondary)] font-semibold whitespace-nowrap">
                            Amount {resultType === 'won' ? 'won' : 'lost'}:
                        </label>
                        <input type="number" value={resultAmt} onChange={e => setResultAmt(e.target.value)}
                            placeholder="Chips" autoFocus
                            className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-3 py-3 text-lg text-white font-bold text-center" />
                    </div>
                    <button onClick={() => { if (resultAmt) onConfirm(resultType === 'won', Number(resultAmt)); }}
                        disabled={!resultAmt}
                        className="w-full py-4 rounded-2xl font-bold text-sm uppercase tracking-wider transition-all duration-200 bg-[var(--color-accent)] text-slate-950 hover:bg-emerald-400 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_4px_16px_rgba(16,185,129,0.3)]">
                        Confirm & Next Hand
                    </button>
                </div>
            )}

            <button onClick={onUndo}
                className="text-sm text-[var(--color-text-secondary)] hover:text-white transition-colors">
                ↩ Back to hand
            </button>
        </div>
    );
}
