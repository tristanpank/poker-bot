'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const ArrowLeftIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-6 h-6"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
);

const UsersIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-8 h-8"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
);

export default function JoinPage() {
  const [code, setCode] = useState('');
  const router = useRouter();

  const handleJoin = (e: React.FormEvent) => {
    e.preventDefault();
    if (code.trim().length > 0) {
      router.push(`/room/${code.trim()}`);
    }
  };

  return (
    <div className="design-page-wrapper">
      <div className="design-mobile-container">
        <div className="w-full max-w-[480px] min-h-screen flex flex-col items-center justify-center p-6 relative [background:radial-gradient(circle_at_top_right,rgba(59,130,246,0.1),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(59,130,246,0.05),transparent_40%)]">
          <Link 
            href="/" 
            className="absolute top-6 left-6 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors p-2 rounded-full hover:bg-[var(--color-surface)]"
            aria-label="Back to home"
          >
            <ArrowLeftIcon className="w-6 h-6" />
          </Link>
          
          <div className="text-center animate-slide-up" style={{ animationDelay: '100ms', animationFillMode: 'forwards' }}>
            <div className="flex justify-center mb-6">
              <div className="bg-[var(--color-info)]/20 p-4 rounded-full border border-[var(--color-info)]/50 shadow-[0_0_20px_rgba(59,130,246,0.4)]">
                <UsersIcon className="w-8 h-8 text-[var(--color-info)]" />
              </div>
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-[var(--color-text-primary)] mb-2">Join Table</h1>
            <p className="text-[var(--color-text-secondary)] text-sm tracking-[0.1em] uppercase font-semibold mb-8">Enter the host's room code</p>
          </div>

          <form onSubmit={handleJoin} className="w-full max-w-sm flex flex-col gap-6 animate-slide-up" style={{ animationDelay: '200ms', animationFillMode: 'forwards' }}>
            <div>
              <label htmlFor="code" className="sr-only">Room Code</label>
              <input
                id="code"
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
                placeholder="ABCD-1234"
                className="w-full bg-[var(--color-surface)] border-2 border-[var(--color-border-color)] rounded-2xl px-6 py-5 text-[var(--color-text-primary)] text-center text-2xl font-mono tracking-widest focus:outline-none focus:border-[var(--color-info)] focus:shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all placeholder:text-[var(--color-text-secondary)]/50 uppercase"
                autoFocus
                maxLength={10}
              />
            </div>
            
            <button
              type="submit"
              disabled={!code.trim()}
              className="w-full py-5 px-6 rounded-2xl font-bold text-lg tracking-[0.1em] uppercase transition-all duration-300 bg-[var(--color-info)] text-white hover:bg-blue-400 hover:shadow-[0_0_30px_rgba(59,130,246,0.6)] active:scale-95 disabled:bg-[var(--color-surface)] disabled:text-[var(--color-text-secondary)] disabled:border disabled:border-[var(--color-border-color)] disabled:shadow-none disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-3"
            >
              Join <span className="text-xl">→</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
