'use client';

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="design-page-wrapper">
      <div className="design-mobile-container flex flex-col items-center justify-center px-4 md:px-6 relative overflow-hidden h-full">
        {/* Subtle decorative background elements */}
        <div className="pointer-events-none absolute -top-40 -left-40 h-80 w-80 rounded-full bg-emerald-500/10 blur-[100px]" />
        <div className="pointer-events-none absolute -bottom-40 -right-40 h-80 w-80 rounded-full bg-blue-500/10 blur-[100px]" />

        <main className="w-full flex flex-col gap-8 animate-fade-in relative z-10 pb-10">
          
          {/* Header Section */}
          <section className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-semibold uppercase tracking-widest text-emerald-400">
                Poker Bot AI
              </span>
            </div>
            <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent pb-1">
              Welcome
            </h1>
            <p className="text-sm text-slate-400 max-w-xs mx-auto leading-relaxed">
              Experience next-level poker analysis. Create a new bot operating session or join an existing game to connect your cam.
            </p>
          </section>

          {/* Action Buttons Section */}
          <section className="flex flex-col gap-4 mt-2">
            <Link href="/play" className="group relative w-full active:scale-[0.98] transition-transform">
              <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-500 opacity-20 blur transition duration-300 group-hover:opacity-40" />
              <div className="relative flex items-center justify-between rounded-xl border border-emerald-500/30 bg-slate-900/90 p-5 shadow-2xl backdrop-blur-sm transition-all duration-300 hover:bg-slate-800 hover:border-emerald-500/50">
                <div className="flex flex-col pr-4 text-left">
                  <span className="text-lg font-bold text-white mb-0.5 tracking-wide">Create Game</span>
                  <span className="text-xs text-slate-400 font-medium tracking-wide leading-snug">Start a new operating session</span>
                </div>
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400 transition-colors duration-300 group-hover:bg-emerald-500 group-hover:text-slate-950 group-hover:shadow-[0_0_15px_rgba(16,185,129,0.5)]">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                  </svg>
                </div>
              </div>
            </Link>

            <Link href="/join" className="group relative w-full active:scale-[0.98] transition-transform">
              <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-blue-500 to-indigo-500 opacity-20 blur transition duration-300 group-hover:opacity-40" />
              <div className="relative flex items-center justify-between rounded-xl border border-blue-500/30 bg-slate-900/90 p-5 shadow-2xl backdrop-blur-sm transition-all duration-300 hover:bg-slate-800 hover:border-blue-500/50">
                <div className="flex flex-col pr-4 text-left">
                  <span className="text-lg font-bold text-white mb-0.5 tracking-wide">Join Game</span>
                  <span className="text-xs text-slate-400 font-medium tracking-wide leading-snug">Connect opponent webcam</span>
                </div>
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-blue-500/10 text-blue-400 transition-colors duration-300 group-hover:bg-blue-500 group-hover:text-slate-950 group-hover:shadow-[0_0_15px_rgba(59,130,246,0.5)]">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
              </div>
            </Link>
          </section>

        </main>
      </div>
    </div>
  );
}
