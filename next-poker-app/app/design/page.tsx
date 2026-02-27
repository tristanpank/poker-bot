'use client';

import { useState, useEffect } from 'react';
import { TextField, InputAdornment, IconButton, ThemeProvider, createTheme } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import Link from 'next/link';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#10b981', // emerald-500
    },
  },
});

function PokerGame() {
  const [mounted, setMounted] = useState(false);
  const [activeAction, setActiveAction] = useState<string | null>(null);
  const [isCamOpen, setIsCamOpen] = useState(false);
  const [tableSize, setTableSize] = useState<number>(6);
  const [setupStep, setSetupStep] = useState<'size' | 'details' | 'game'>('size');
  const [smallBlind, setSmallBlind] = useState<number>(1);
  const [bigBlind, setBigBlind] = useState<number>(2);
  const [buyIn, setBuyIn] = useState<number>(200);
  const [spotsFromBB, setSpotsFromBB] = useState<number>(0);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleAction = (action: string) => {
    setActiveAction(action);
    setTimeout(() => setActiveAction(null), 300);
  };

  if (!mounted) return null;

  if (setupStep === 'size') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-10 p-6 min-h-[100dvh]">
        <div className="text-center animate-slide-up [animation-delay:100ms] [animation-fill-mode:forwards]">
          <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-3 flex items-center justify-center gap-3">
            <span className="text-5xl text-[var(--color-accent)]">♠</span>
            PokerBot
          </h1>
          <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.2em] uppercase font-semibold">Select Table Size</p>
          <Link href="/">
            <button
              type="button"
              className="rounded-md bg-rose-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-rose-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
            >
              Go to Computer Vision Test
            </button>
          </Link>
        </div>

        <div className="w-full max-w-sm grid grid-cols-2 gap-4 animate-slide-up [animation-delay:200ms] [animation-fill-mode:forwards]">
          {[2, 3, 4, 5, 6].map((size) => (
            <button
              key={size}
              onClick={() => setTableSize(size)}
              className={`p-6 rounded-2xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-1 ${tableSize === size
                ? 'bg-[var(--color-accent)]/15 border-[var(--color-accent)] shadow-[0_0_20px_var(--color-accent-glow)] text-[var(--color-accent)] scale-105'
                : 'bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)] hover:border-[var(--color-text-secondary)]'
                }`}
            >
              <span className="text-4xl font-bold">{size}</span>
              <span className={`text-[10px] uppercase tracking-wider font-semibold ${tableSize === size ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-secondary)]'}`}>Max Players</span>
            </button>
          ))}
        </div>

        <div className="w-full max-w-sm mt-6 animate-slide-up [animation-delay:300ms] [animation-fill-mode:forwards]">
          <button
            onClick={() => setSetupStep('details')}
            className="w-full py-5 px-6 rounded-2xl font-bold text-lg tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-3"
          >
            Continue
            <span className="text-xl">→</span>
          </button>
        </div>
      </div>
    );
  }

  if (setupStep === 'details') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-8 p-6 min-h-[100dvh]">
        <div className="text-center animate-slide-up [animation-delay:100ms] [animation-fill-mode:forwards]">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-2 flex items-center justify-center gap-3">
            Game Details
          </h1>
          <p className="text-[var(--color-text-secondary)] text-xs tracking-[0.2em] uppercase font-semibold">Configure Your Session</p>
        </div>

        <div className="w-full max-w-sm flex flex-col gap-5 animate-slide-up [animation-delay:200ms] [animation-fill-mode:forwards]">

          <div className="bg-[var(--color-surface)] border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] flex flex-col gap-5">
            <div className="flex justify-between items-center">
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Blinds (SB / BB)</label>
              <div className="flex items-center gap-2">
                <TextField
                  type="number"
                  value={smallBlind}
                  onChange={(e) => setSmallBlind(Number(e.target.value))}
                  size="small"
                  sx={{ width: 70 }}
                  inputProps={{ style: { textAlign: 'center', fontWeight: 'bold' } }}
                />
                <span className="text-[var(--color-text-secondary)] font-bold">/</span>
                <TextField
                  type="number"
                  value={bigBlind}
                  onChange={(e) => setBigBlind(Number(e.target.value))}
                  size="small"
                  sx={{ width: 70 }}
                  inputProps={{ style: { textAlign: 'center', fontWeight: 'bold' } }}
                />
              </div>
            </div>

            <div className="h-[1px] w-full bg-[var(--color-border-color)]" />

            <div className="flex justify-between items-center">
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Buy In</label>
              <TextField
                type="number"
                value={buyIn}
                onChange={(e) => setBuyIn(Number(e.target.value))}
                size="small"
                sx={{ width: 120 }}
                InputProps={{
                  startAdornment: <InputAdornment position="start">$</InputAdornment>,
                  style: { fontWeight: 'bold' }
                }}
              />
            </div>

            <div className="h-[1px] w-full bg-[var(--color-border-color)]" />

            <div className="flex justify-between items-center">
              <label className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider font-semibold">Spots from BB</label>
              <TextField
                type="number"
                value={spotsFromBB}
                onChange={(e) => setSpotsFromBB(Number(e.target.value))}
                size="small"
                sx={{ width: 80 }}
                inputProps={{ style: { textAlign: 'center', fontWeight: 'bold' } }}
              />
            </div>
          </div>
        </div>

        <div className="w-full max-w-sm flex gap-3 mt-4 animate-slide-up [animation-delay:300ms] [animation-fill-mode:forwards]">
          <button
            onClick={() => setSetupStep('size')}
            className="flex-1 py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 border border-[var(--color-border-color)] bg-[var(--color-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text-primary)] active:scale-95 flex items-center justify-center gap-2"
          >
            <span className="text-lg">←</span>
            Back
          </button>
          <button
            onClick={() => setSetupStep('game')}
            className="flex-[2] py-4 px-4 rounded-2xl font-bold text-sm tracking-[0.1em] uppercase transition-all duration-300 shadow-[0_4px_24px_rgba(16,185,129,0.25)] bg-[var(--color-accent)] text-slate-950 hover:shadow-[0_0_30px_var(--color-accent-glow)] hover:bg-emerald-400 active:scale-95 flex items-center justify-center gap-2"
          >
            Initialize Agent
            <span className="text-lg">→</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <header className="p-6 flex justify-between items-center backdrop-blur-md border-b border-[var(--color-border-color)] bg-slate-950/30 sticky top-0 z-10 animate-slide-up [animation-delay:100ms] [animation-fill-mode:forwards]">
        <div className="flex items-center gap-3">
          <IconButton
            onClick={() => setSetupStep('details')}
            sx={{
              width: 36, height: 36, border: '1px solid var(--color-border-color)',
              color: 'var(--color-text-secondary)',
              '&:hover': { color: 'var(--color-text-primary)', bgcolor: 'var(--color-surface-hover)' }
            }}
            title="Back to Details"
            size="small"
          >
            <ArrowBackIcon fontSize="small" />
          </IconButton>
          <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent flex items-center gap-2">
            <span className="text-xl text-[var(--color-accent)]">♠</span>
            PokerBot
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            className={`w-9 h-9 rounded-full flex items-center justify-center cursor-pointer text-xl transition-all duration-200 ${isCamOpen
              ? 'bg-emerald-500/20 border border-[var(--color-accent)] shadow-[0_0_12px_var(--color-accent-glow)]'
              : 'bg-[var(--color-surface)] border border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-hover)]'
              }`}
            onClick={() => setIsCamOpen(!isCamOpen)}
            title="Toggle Agent Vision"
          >
            {isCamOpen ? '👁️' : '👁️‍🗨️'}
          </button>
          <div className="inline-flex items-center gap-1.5 text-sm text-[var(--color-accent)] font-medium px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <div className="w-2 h-2 rounded-full bg-[var(--color-accent)] shadow-[0_0_8px_var(--color-accent-glow)] animate-[var(--animate-pulse-accent)]"></div>
            Live
          </div>
        </div>
      </header>

      {/* Cam Window */}
      <div
        className={`absolute top-[80px] right-6 w-40 h-[120px] bg-black border-2 border-[var(--color-border-color)] rounded-xl overflow-hidden z-50 shadow-[0_8px_32px_rgba(0,0,0,0.5)] ${isCamOpen ? 'block animate-[var(--animate-slide-in-right)]' : 'hidden'}`}
      >
        <div className="absolute inset-0 [background-size:100%_4px] pointer-events-none"></div>
        <div className="absolute top-1 right-1.5 text-[var(--color-danger)] text-[8px] flex items-center gap-1 font-bold">
          <div className="w-[6px] h-[6px] bg-[var(--color-danger)] rounded-full shadow-[0_0_4px_var(--color-danger-glow)] animate-[var(--animate-pulse-danger)]"></div>
          REC
        </div>
        <div className="w-full h-full bg-slate-950 flex items-center justify-center flex-col gap-1">
        </div>
      </div>

      <main className="flex-1 p-6 flex flex-col gap-6 overflow-y-auto pb-[120px]">
        {/* Recommendation Panel */}
        <section className="bg-[var(--color-surface)] backdrop-blur-md border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] transition-all duration-200 hover:bg-[var(--color-surface-hover)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] animate-slide-up [animation-delay:200ms] [animation-fill-mode:forwards]">
          <div className="flex flex-col items-center justify-center py-8 px-4 text-center min-h-[160px]">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-[0.05em] font-semibold mb-4">Agent Recommendation</span>

            <div className="flex justify-center -space-x-2.5 relative mb-6">
              <div className="w-[60px] h-[84px] bg-white rounded-md flex flex-col justify-center items-center text-2xl font-bold shadow-[0_4px_12px_rgba(0,0,0,0.5)] border border-white/80 relative transition-transform duration-300 -rotate-6 -translate-x-2.5 z-10 text-slate-900">
                <span className="absolute top-1 left-1.5 text-base leading-none">10</span>
                <span className="text-3xl">♠</span>
              </div>
              <div className="w-[60px] h-[84px] bg-white rounded-md flex flex-col justify-center items-center text-2xl font-bold shadow-[0_4px_12px_rgba(0,0,0,0.5)] border border-white/80 relative transition-transform duration-300 rotate-6 translate-x-2.5 text-red-500">
                <span className="absolute top-1 left-1.5 text-base leading-none">K</span>
                <span className="text-3xl">♥</span>
              </div>
            </div>

            <div className="text-4xl font-bold uppercase tracking-[0.02em] mb-2 text-[var(--color-accent)]" style={{ textShadow: '0 0 20px var(--color-accent-glow)' }}>
              RAISE 3BB
            </div>

            <div className="w-full mt-2">
              <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
                <span>Confidence</span>
                <span className="text-[var(--color-accent)] font-semibold">92%</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden mt-4">
                <div className="h-full bg-[var(--color-accent)] shadow-[0_0_10px_var(--color-accent-glow)] rounded-full transition-all duration-500 ease-out" style={{ width: '92%' }}></div>
              </div>
            </div>
          </div>
        </section>

        {/* Stats Grid */}
        <section className="grid grid-cols-2 gap-4 animate-slide-up [animation-delay:300ms] [animation-fill-mode:forwards]">
          <div className="bg-[var(--color-surface)] backdrop-blur-md border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] transition-all duration-200 hover:bg-[var(--color-surface-hover)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-[0.05em] font-semibold">Hand Strength</span>
            <span className="text-2xl font-bold text-[var(--color-accent)]">High</span>
          </div>
          <div className="bg-[var(--color-surface)] backdrop-blur-md border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] transition-all duration-200 hover:bg-[var(--color-surface-hover)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-[0.05em] font-semibold">Pot Odds</span>
            <span className="text-2xl font-bold text-[var(--color-text-primary)]">32%</span>
          </div>
          <div className="bg-[var(--color-surface)] backdrop-blur-md border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] transition-all duration-200 hover:bg-[var(--color-surface-hover)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-[0.05em] font-semibold">Expected Value</span>
            <span className="text-2xl font-bold text-[var(--color-info)]">+1.5BB</span>
          </div>
          <div className="bg-[var(--color-surface)] backdrop-blur-md border border-[var(--color-border-color)] rounded-2xl p-5 shadow-[0_4px_24px_rgba(0,0,0,0.2)] transition-all duration-200 hover:bg-[var(--color-surface-hover)] hover:-translate-y-0.5 hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-[0.05em] font-semibold">Opponent Profile</span>
            <span className="text-xl font-bold text-[var(--color-text-primary)]">LAG</span>
          </div>
        </section>
      </main>

      <nav className="fixed bottom-0 w-full max-w-[480px] p-6 bg-gradient-to-t from-slate-950 from-60% to-transparent grid grid-cols-3 gap-3 z-20 animate-slide-up [animation-delay:400ms] [animation-fill-mode:forwards]">
        <button
          className={`border-none font-inherit text-lg font-semibold rounded-2xl py-5 px-2 cursor-pointer flex flex-col items-center justify-center gap-1 transition-all duration-200 uppercase tracking-[0.05em] relative overflow-hidden active:scale-95 after:content-[''] after:absolute after:inset-0 after:bg-gradient-to-b after:from-white/15 after:to-transparent after:pointer-events-none text-[var(--color-danger)] border border-[rgba(239,68,68,0.3)] ${activeAction === 'fold'
            ? 'bg-[var(--color-danger)] !text-white shadow-[0_0_20px_var(--color-danger-glow)]'
            : 'bg-red-500/10 hover:bg-[var(--color-danger)] hover:text-white hover:shadow-[0_0_20px_var(--color-danger-glow)]'
            }`}
          onClick={() => handleAction('fold')}
        >
          <span className="text-2xl mb-1">⨯</span>
          Fold
        </button>
        <button
          className={`border-none font-inherit text-lg font-semibold rounded-2xl py-5 px-2 cursor-pointer flex flex-col items-center justify-center gap-1 transition-all duration-200 uppercase tracking-[0.05em] relative overflow-hidden active:scale-95 after:content-[''] after:absolute after:inset-0 after:bg-gradient-to-b after:from-white/15 after:to-transparent after:pointer-events-none text-[var(--color-info)] border border-[rgba(59,130,246,0.3)] ${activeAction === 'call'
            ? 'bg-[var(--color-info)] !text-white shadow-[0_0_20px_rgba(59,130,246,0.5)]'
            : 'bg-blue-500/10 hover:bg-[var(--color-info)] hover:text-white hover:shadow-[0_0_20px_rgba(59,130,246,0.5)]'
            }`}
          onClick={() => handleAction('call')}
        >
          <span className="text-2xl mb-1">−</span>
          Call
        </button>
        <button
          className={`border-none font-inherit text-lg font-semibold rounded-2xl py-5 px-2 cursor-pointer flex flex-col items-center justify-center gap-1 transition-all duration-200 uppercase tracking-[0.05em] relative overflow-hidden active:scale-95 after:content-[''] after:absolute after:inset-0 after:bg-gradient-to-b after:from-white/15 after:to-transparent after:pointer-events-none text-[var(--color-accent)] border border-[rgba(16,185,129,0.3)] ${activeAction === 'raise'
            ? 'bg-[var(--color-accent)] !text-white shadow-[0_0_20px_var(--color-accent-glow)]'
            : 'bg-emerald-500/10 hover:bg-[var(--color-accent)] hover:text-white hover:shadow-[0_0_20px_var(--color-accent-glow)]'
            }`}
          onClick={() => handleAction('raise')}
        >
          <span className="text-2xl mb-1">↑</span>
          Raise
        </button>
      </nav>
    </>
  );
}

export default function DesignPage() {
  return (
    <ThemeProvider theme={darkTheme}>
      <PokerGame />
    </ThemeProvider>
  );
}
