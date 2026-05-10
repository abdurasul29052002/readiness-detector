"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import NotificationBell from "./NotificationBell";
import ModelPanel from "./ModelPanel";

const NAV_ITEMS = [
  { href: "/", label: "Aniqlash" },
  { href: "/statistics", label: "Statistika", auth: true },
  { href: "/cameras", label: "Kameralar", auth: true },
  { href: "/video", label: "Video tahlil", auth: true },
  { href: "/users", label: "Foydalanuvchilar", admin: true },
];

export default function HeaderNav() {
  const { user, isAuthenticated, isAdmin, logout } = useAuth();
  const pathname = usePathname();

  return (
    <div className="flex items-center gap-3">
      <nav className="hidden md:flex items-center gap-1">
        {NAV_ITEMS.map((item) => {
          if (item.admin && !isAdmin) return null;
          if (item.auth && !isAuthenticated) return null;
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                active
                  ? "bg-zinc-200 text-zinc-900"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full glass text-xs font-semibold text-zinc-600">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        AI Server
      </div>

      <ModelPanel />
      <NotificationBell />

      {isAuthenticated ? (
        <div className="flex items-center gap-2">
          <span className="px-3 py-1.5 rounded-full glass text-xs font-semibold text-zinc-600">
            {user?.full_name}
            <span className={`ml-1.5 px-1.5 py-0.5 rounded text-[10px] ${
              isAdmin ? "bg-indigo-500/15 text-indigo-700" : "bg-emerald-500/15 text-emerald-700"
            }`}>
              {user?.role}
            </span>
          </span>
          <button
            onClick={logout}
            className="px-3 py-1.5 rounded-full glass text-xs font-semibold text-red-600 hover:text-red-700 hover:bg-red-500/10 transition-all cursor-pointer"
          >
            Chiqish
          </button>
        </div>
      ) : (
        <Link
          href="/login"
          className="px-4 py-1.5 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 text-xs font-semibold text-white hover:-translate-y-0.5 transition-all"
        >
          Kirish
        </Link>
      )}
    </div>
  );
}
