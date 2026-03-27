import { redirect } from "next/navigation";

import Link from "next/link";

const PlusCircleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
);

const UsersIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
);

export default function HomePage() {
  return (
    <div className="design-page-wrapper">
      <div className="design-mobile-container">
        <div className="w-full max-w-[480px] min-h-screen flex flex-col items-center justify-center p-6 relative [background:radial-gradient(circle_at_top_right,rgba(59,130,246,0.1),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.05),transparent_40%)]">
          <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
            <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-text-primary)] to-[var(--color-text-secondary)] bg-clip-text text-transparent mb-3 flex items-center justify-center gap-3">
              <span className="text-5xl text-[var(--color-accent)]">♠</span>
              PokerBot
            </h1>
            <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.2em] uppercase font-semibold mb-12">Ready to hit the tables?</p>
          </div>
          
          <div className="w-full max-w-sm flex flex-col gap-4 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
            <Link
              href="/play"
              className="group p-6 rounded-2xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-2 bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-accent)]/15 hover:border-[var(--color-accent)] hover:shadow-[0_0_20px_var(--color-accent-glow)] hover:text-[var(--color-accent)]"
            >
              <div className="flex items-center gap-2 text-xl font-bold uppercase tracking-wider">
                <PlusCircleIcon />
                <span>Create Game</span>
              </div>
              <span className="text-xs text-[var(--color-text-secondary)] font-semibold uppercase tracking-widest group-hover:text-[var(--color-accent)]/80">Host a new table</span>
            </Link>

            <Link
              href="/join"
              className="group p-6 rounded-2xl border-2 transition-all duration-300 flex flex-col items-center justify-center gap-2 bg-[var(--color-surface)] border-[var(--color-border-color)] text-[var(--color-text-primary)] hover:bg-[var(--color-info)]/15 hover:border-[var(--color-info)] hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] hover:text-[var(--color-info)]"
            >
              <div className="flex items-center gap-2 text-xl font-bold uppercase tracking-wider">
                <UsersIcon />
                <span>Join Game</span>
              </div>
              <span className="text-xs text-[var(--color-text-secondary)] font-semibold uppercase tracking-widest group-hover:text-[var(--color-info)]/80">Enter room code</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
