'use client';

import { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

const ArrowLeftIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-5 h-5"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
);

const CameraIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-5 h-5"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
);

const CameraOffIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-5 h-5"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="2" y1="2" x2="22" y2="22"/><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9c0 1.1.9 2 2 2h14"/><path d="M22 17V9a2 2 0 0 0-2-2h-3l-2.5-3h-4.5"/><circle cx="12" cy="13" r="3"/></svg>
);

const VideoIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className || "w-16 h-16"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
);

export default function RoomPage() {
  const params = useParams();
  const router = useRouter();
  const code = params.code as string;
  
  const [cameraOn, setCameraOn] = useState(false);

  return (
    <div className="design-page-wrapper">
      <div className="design-mobile-container">
        <div className="w-full max-w-[480px] min-h-[100dvh] flex flex-col text-[var(--color-text-primary)] relative [background:radial-gradient(circle_at_top_right,rgba(59,130,246,0.1),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.05),transparent_40%)]">
          <header className="flex items-center justify-between p-6 animate-slide-up" style={{ animationDelay: '50ms', animationFillMode: 'forwards' }}>
            <Link 
              href="/join" 
              className="flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors p-2 -ml-2 rounded-full hover:bg-[var(--color-surface)]"
            >
              <ArrowLeftIcon className="w-5 h-5" />
              <span className="font-semibold tracking-[0.1em] uppercase text-xs">Leave</span>
            </Link>
            <div className="bg-[var(--color-surface)] border border-[var(--color-accent)]/50 px-4 py-2 rounded-full font-mono text-[var(--color-accent)] tracking-widest text-sm shadow-[0_0_10px_var(--color-accent-glow)]">
              {code?.toUpperCase()}
            </div>
          </header>

          <main className="flex-1 flex flex-col items-center justify-center p-6 w-full gap-8">
            <div className="text-center animate-slide-up" style={{ animationDelay: '150ms', animationFillMode: 'forwards' }}>
              <h1 className="text-3xl font-bold mb-2 tracking-tight">Waiting for Host</h1>
              <p className="text-[var(--color-text-secondary)] text-sm uppercase tracking-widest font-semibold flex items-center justify-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[var(--color-warning)] animate-pulse"></span>
                Game starting soon
              </p>
            </div>

            {/* Camera Placeholder Area */}
            <div className="w-full aspect-[3/4] max-h-[60vh] bg-black/40 border-2 border-[var(--color-border-color)] rounded-3xl overflow-hidden relative shadow-[0_10px_40px_rgba(0,0,0,0.5)] transition-all duration-500 animate-slide-up" style={{ animationDelay: '250ms', animationFillMode: 'forwards' }}>
              {cameraOn ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-surface)]">
                  <VideoIcon className="w-16 h-16 text-[var(--color-info)] mb-4 animate-[pulse-accent_2s_infinite]" />
                  <p className="text-[var(--color-info)] font-bold tracking-widest uppercase text-sm">Feed Active</p>
                </div>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60">
                  <CameraOffIcon className="w-16 h-16 text-[var(--color-text-secondary)] mb-4 opacity-50" />
                  <p className="text-[var(--color-text-secondary)] font-bold tracking-widest uppercase text-sm">Camera Off</p>
                </div>
              )}
              
              {/* Controls Bar */}
              <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 via-black/50 to-transparent flex justify-center">
                <button
                  onClick={() => setCameraOn(!cameraOn)}
                  className={`flex items-center gap-2 px-8 py-4 rounded-full font-bold tracking-wider uppercase text-sm transition-all shadow-lg ${
                    cameraOn 
                      ? 'bg-[var(--color-danger)]/20 text-[var(--color-danger)] border border-[var(--color-danger)]/50 hover:bg-[var(--color-danger)]/40 hover:shadow-[0_0_20px_var(--color-danger-glow)]' 
                      : 'bg-[var(--color-accent)]/20 text-[var(--color-accent)] border border-[var(--color-accent)]/50 hover:bg-[var(--color-accent)]/40 hover:shadow-[0_0_20px_var(--color-accent-glow)]'
                  }`}
                >
                  {cameraOn ? (
                    <>
                      <CameraOffIcon className="w-5 h-5" />
                      <span>Disable</span>
                    </>
                  ) : (
                    <>
                      <CameraIcon className="w-5 h-5" />
                      <span>Enable</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
