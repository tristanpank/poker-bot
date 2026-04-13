import React from 'react';
import { sixSeatLayout } from '../../lib/tablePositions';

type SeatTone = 'bot' | 'connected' | 'manual' | 'active' | 'normal' | 'folded' | 'open';

export type TableSeatVisual = {
    seat: number;
    title: string;
    subtitle: string;
    detail?: string | null;
    tone: SeatTone;
    onClick?: (() => void) | null;
    onDealerDrop?: (() => void) | null;
    onDealerDragStart?: (() => void) | null;
    onDealerDragEnd?: (() => void) | null;
    disabled?: boolean;
    isDealer?: boolean;
    isBot?: boolean;
    canAcceptDealerDrop?: boolean;
    dealerDraggable?: boolean;
};

type TableVisualProps = {
    seats: TableSeatVisual[];
    center: React.ReactNode;
};

function seatToneClass(tone: SeatTone, disabled: boolean): string {
    if (disabled) {
        return 'border-slate-700/40 bg-slate-900/70 text-slate-500 opacity-60';
    }
    switch (tone) {
        case 'bot':
            return 'border-emerald-400 bg-emerald-500/20 text-emerald-100 shadow-[0_0_30px_rgba(16,185,129,0.2)]';
        case 'connected':
            return 'border-sky-400/40 bg-sky-500/10 text-slate-100';
        case 'manual':
            return 'border-amber-400/50 bg-amber-500/10 text-amber-100';
        case 'active':
            return 'border-[var(--color-accent)] bg-[var(--color-accent)]/15 text-[var(--color-text-primary)] shadow-[0_0_28px_rgba(16,185,129,0.15)]';
        case 'folded':
            return 'border-slate-700/40 bg-slate-900/60 text-slate-400 opacity-70';
        case 'normal':
            return 'border-[var(--color-border-color)] bg-slate-950/55 text-[var(--color-text-primary)]';
        default:
            return 'border-[var(--color-border-color)] bg-slate-950/45 text-[var(--color-text-primary)]';
    }
}

export default function TableVisual({ seats, center }: TableVisualProps) {
    const seatMap = new Map(seats.map((seat) => [seat.seat, seat]));

    return (
        <section className="w-full rounded-[2rem] border border-[var(--color-border-color)] bg-[var(--color-surface)]/85 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.35)]">
            <div className="relative mx-auto h-[28rem] max-w-4xl">
                <div className="absolute inset-4 rounded-[999px] border border-emerald-500/15 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.18),rgba(2,6,23,0.2)_60%,rgba(2,6,23,0.92)_100%)] shadow-[inset_0_0_40px_rgba(16,185,129,0.12)]" />
                <div className="absolute inset-[4.5rem] rounded-[999px] border border-white/5 bg-slate-950/35" />

                <div className="absolute inset-[5.25rem] flex items-center justify-center px-6">
                    <div className="w-110 max-w-sm rounded-[2rem] border border-white/8 bg-slate-950/55 px-5 py-4 text-center shadow-[inset_0_0_30px_rgba(15,23,42,0.45)] backdrop-blur-sm">
                        {center}
                    </div>
                </div>

                {sixSeatLayout.map(({ seat, className }) => {
                    const data = seatMap.get(seat) ?? {
                        seat,
                        title: `Seat ${seat + 1}`,
                        subtitle: 'Open',
                        detail: null,
                        tone: 'open' as const,
                        onClick: null,
                        onDealerDrop: null,
                        onDealerDragStart: null,
                        onDealerDragEnd: null,
                        disabled: false,
                        canAcceptDealerDrop: false,
                        dealerDraggable: false,
                    };
                    const disabled = Boolean(data.disabled);
                    const clickable = typeof data.onClick === 'function' && !disabled;
                    const droppable = typeof data.onDealerDrop === 'function';
                    const dragTarget = Boolean(data.canAcceptDealerDrop);

                    return (
                        <button
                            key={seat}
                            type="button"
                            onClick={() => {
                                if (clickable) {
                                    data.onClick?.();
                                }
                            }}
                            onDragOver={droppable ? (event) => {
                                event.preventDefault();
                                event.dataTransfer.dropEffect = 'move';
                            } : undefined}
                            onDrop={droppable ? (event) => {
                                event.preventDefault();
                                data.onDealerDrop?.();
                            } : undefined}
                            aria-disabled={disabled}
                            className={`absolute ${className} flex h-24 w-24 flex-col items-center justify-center rounded-3xl border text-center transition-all duration-200 ${seatToneClass(data.tone, disabled)} ${clickable ? 'hover:scale-[1.03]' : ''} ${droppable ? 'cursor-pointer' : ''} ${dragTarget ? 'ring-2 ring-amber-300 ring-offset-2 ring-offset-slate-950' : ''}`}
                        >
                            {data.isDealer && (
                                <span
                                    draggable={Boolean(data.dealerDraggable)}
                                    onDragStart={(event) => {
                                        event.dataTransfer.effectAllowed = 'move';
                                        event.dataTransfer.setData('text/plain', 'dealer-button');
                                        data.onDealerDragStart?.();
                                    }}
                                    onDragEnd={() => {
                                        data.onDealerDragEnd?.();
                                    }}
                                    className={`absolute -top-2 -right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full border-2 border-slate-900 bg-white text-[9px] font-black text-slate-900 shadow-md ${data.dealerDraggable ? 'cursor-grab active:cursor-grabbing' : ''}`}
                                    title="Dealer Button"
                                >
                                    D
                                </span>
                            )}
                            {(data.isBot || data.tone === 'bot') && (
                                <span
                                    className="absolute -top-2 -left-2 z-10 flex h-6 w-6 items-center justify-center rounded-full border-2 border-slate-900 bg-amber-400 text-[11px] shadow-[0_0_8px_rgba(251,191,36,0.6)]"
                                    title="Host"
                                >
                                    👑
                                </span>
                            )}
                            <span className="text-[10px] font-bold uppercase tracking-[0.18em]">{data.title}</span>
                            <span className="mt-1 px-1 text-[11px] font-semibold text-white/90">{data.subtitle}</span>
                            <span className="mt-1 px-1 text-[9px] uppercase tracking-[0.15em] text-white/60">
                                {data.detail ?? ''}
                            </span>
                        </button>
                    );
                })}
            </div>
        </section>
    );
}
