import type { Metadata, Viewport } from 'next';

export const metadata: Metadata = {
    title: 'PokerBot — Play',
    description: 'Mobile-first poker bot assistant for live table play',
};

export const viewport: Viewport = {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
};

export default function PlayLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="design-page-wrapper">
            <div className="design-mobile-container">
                <div className="w-full max-w-[480px] min-h-screen flex flex-col relative [background:radial-gradient(circle_at_top_right,rgba(59,130,246,0.1),transparent_40%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.05),transparent_40%)]">
                    {children}
                </div>
            </div>
        </div>
    );
}
