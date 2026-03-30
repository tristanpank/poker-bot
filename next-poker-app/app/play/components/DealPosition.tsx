import React from 'react';
import {
    compactSeatMap,
    getCompactRoleForSeat,
    getDefaultPlayerName,
    getSeatLabel,
    sixSeatLayout,
} from '../../lib/tablePositions';

type DealPositionProps = {
    botSeat: number | null;
    connectedSeats: number[];
    manualSeats: number[];
    seatNames: Record<string, string>;
    onSeatClick: (seatIndex: number) => void;
    onContinue: () => void;
    canContinue: boolean;
    continueLabel: string;
    statusMessage: string;
    canUndo: boolean;
    onUndo: () => void;
};

export default function DealPosition({
    botSeat,
    connectedSeats,
    manualSeats,
    seatNames,
    onSeatClick,
    onContinue,
    canContinue,
    continueLabel,
    statusMessage,
    canUndo,
    onUndo,
}: DealPositionProps) {
    const occupiedSeats = compactSeatMap(botSeat === null ? [...connectedSeats, ...manualSeats] : [botSeat, ...connectedSeats, ...manualSeats]);

    return (
        <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
            <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
                <h1 className="text-3xl font-bold tracking-tight text-[var(--color-text-primary)] mb-2">Table Seats</h1>
                <p className="text-[var(--color-text-secondary)] text-xs tracking-[0.2em] uppercase font-semibold">Pick the bot seat, then click empty seats to mark manual players</p>
            </div>

            <div className="w-full max-w-xl rounded-[2rem] border border-[var(--color-border-color)] bg-[var(--color-surface)]/80 p-4 shadow-[0_24px_80px_rgba(0,0,0,0.35)] animate-slide-up" style={{ animationDelay: '180ms', animationFillMode: 'forwards' }}>
                <div className="relative mx-auto h-[22rem] max-w-lg">
                    <div className="absolute inset-10 rounded-[999px] border border-emerald-500/20 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.18),rgba(2,6,23,0.2)_60%,rgba(2,6,23,0.92)_100%)] shadow-[inset_0_0_40px_rgba(16,185,129,0.12)]" />
                    <div className="absolute inset-[4.5rem] rounded-[999px] border border-white/5 bg-slate-950/40" />

                    {sixSeatLayout.map(({ seat, className }) => {
                        const isBotSeat = seat === botSeat;
                        const isConnectedSeat = connectedSeats.includes(seat);
                        const isManualSeat = manualSeats.includes(seat);
                        const role = getCompactRoleForSeat(seat, occupiedSeats);
                        const displayName = seatNames[String(seat)]?.trim() || getDefaultPlayerName(seat);

                        return (
                            <button
                                key={seat}
                                onClick={() => onSeatClick(seat)}
                                className={`absolute ${className} flex h-24 w-24 -translate-y-1/2 flex-col items-center justify-center rounded-3xl border text-center transition-all duration-200 ${
                                    isBotSeat
                                        ? 'border-emerald-400 bg-emerald-500/20 text-emerald-100 shadow-[0_0_30px_rgba(16,185,129,0.2)]'
                                        : isConnectedSeat
                                            ? 'border-sky-400/40 bg-sky-500/10 text-slate-100 hover:border-emerald-400/50 hover:bg-emerald-500/10'
                                            : isManualSeat
                                                ? 'border-amber-400/50 bg-amber-500/10 text-amber-100 hover:border-amber-300 hover:bg-amber-500/15'
                                            : 'border-[var(--color-border-color)] bg-slate-950/50 text-[var(--color-text-primary)] hover:border-[var(--color-accent)] hover:bg-[var(--color-accent)]/10'
                                }`}
                            >
                                <span className="text-xs font-bold uppercase tracking-[0.18em]">{getSeatLabel(seat)}</span>
                                <span className="mt-1 text-[11px] font-semibold text-white/90">
                                    {isBotSeat ? 'Bot' : isConnectedSeat ? displayName : isManualSeat ? 'Manual' : 'Open'}
                                </span>
                                <span className="mt-1 text-[10px] uppercase tracking-[0.16em] text-white/55">
                                    {role ?? (isBotSeat ? 'Waiting' : isConnectedSeat ? 'Webcam' : isManualSeat ? 'Host Seated' : 'Empty')}
                                </span>
                            </button>
                        );
                    })}
                </div>
            </div>

            <div className="w-full max-w-md rounded-2xl border border-[var(--color-border-color)] bg-slate-950/50 px-4 py-3 text-center animate-slide-up" style={{ animationDelay: '260ms', animationFillMode: 'forwards' }}>
                <p className="text-sm font-semibold text-[var(--color-text-primary)]">{statusMessage}</p>
                <p className="mt-1 text-[11px] uppercase tracking-[0.16em] text-[var(--color-text-secondary)]">
                    {occupiedSeats.length} seated
                </p>
            </div>

            <div className="w-full max-w-md flex gap-3 animate-slide-up" style={{ animationDelay: '320ms', animationFillMode: 'forwards' }}>
                {canUndo && (
                    <button onClick={onUndo} className="flex-1 rounded-2xl border border-slate-600 bg-slate-900/70 px-4 py-3 text-sm font-semibold text-slate-300 transition-all hover:bg-slate-800">
                        Undo
                    </button>
                )}
                <button
                    onClick={onContinue}
                    disabled={!canContinue}
                    className="flex-[2] rounded-2xl bg-[var(--color-accent)] px-4 py-3 text-sm font-bold uppercase tracking-[0.14em] text-slate-950 transition-all hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-45"
                >
                    {continueLabel}
                </button>
            </div>
        </div>
    );
}
