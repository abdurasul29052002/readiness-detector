import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import HeaderNav from "@/components/HeaderNav";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EduReady platform",
  description: "EduReady Real-time Student Behavior Monitoring",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="uz" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-screen bg-zinc-50 bg-radial-glow text-zinc-900">
        <AuthProvider>
          {/* Header */}
          <header className="sticky top-0 z-50 border-b border-zinc-200 bg-white/80 backdrop-blur-xl">
            <div className="max-w-[1440px] mx-auto px-8 h-16 flex items-center justify-between">
              <div className="flex items-center gap-3.5">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-lg glow-indigo">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                    <circle cx="12" cy="12" r="3"/>
                  </svg>
                </div>
                <div>
                  <h1 className="text-[17px] font-bold text-gradient tracking-tight">
                    Student Behavior Detection
                  </h1>
                  <p className="text-[11px] text-zinc-500 tracking-wide">
                    YOLOv8 &middot; Real-time Monitoring
                  </p>
                </div>
              </div>

              <HeaderNav />
            </div>
          </header>

          {/* Content */}
          <main className="max-w-[1440px] mx-auto px-8 py-6">
            {children}
          </main>
        </AuthProvider>
      </body>
    </html>
  );
}
