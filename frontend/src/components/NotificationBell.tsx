"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { NotificationItem } from "@/types/detection";
import { getNotifications, markAllNotificationsRead, markNotificationRead } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { useWebSocket } from "@/lib/useWebSocket";

function getWsUrl() {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL + "/ws/notifications";
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/^http/, "ws") + "/ws/notifications";
  }
  if (typeof window === "undefined") return "ws://localhost:8080/ws/notifications";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/notifications`;
}

export default function NotificationBell() {
  const { isAuthenticated } = useAuth();
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [open, setOpen] = useState(false);
  const [toast, setToast] = useState<NotificationItem | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleWsMessage = useCallback((data: unknown) => {
    const notif = data as NotificationItem;
    setNotifications((prev) => [notif, ...prev]);
    setUnreadCount((c) => c + 1);
    setToast(notif);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 5000);
  }, []);

  const { connect, disconnect } = useWebSocket({
    url: getWsUrl(),
    onMessage: handleWsMessage,
    autoReconnect: true,
  });

  useEffect(() => {
    if (isAuthenticated) {
      getNotifications()
        .then((list) => {
          setNotifications(list);
          setUnreadCount(list.filter((n) => !n.read).length);
        })
        .catch(() => {});
      connect();
    }
    return () => disconnect();
  }, [isAuthenticated, connect, disconnect]);

  const handleMarkAllRead = async () => {
    await markAllNotificationsRead();
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    setUnreadCount(0);
  };

  const handleMarkRead = async (id: number) => {
    await markNotificationRead(id);
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read: true } : n)));
    setUnreadCount((c) => Math.max(0, c - 1));
  };

  if (!isAuthenticated) return null;

  return (
    <>
      {/* Toast */}
      {toast && (
        <div className={`fixed top-20 right-8 z-[60] max-w-md px-5 py-3 rounded-xl shadow-2xl border text-sm font-medium animate-[fade-in-up_0.3s_ease] ${
          toast.severity === "CRITICAL"
            ? "bg-red-50 border-red-200 text-red-700"
            : "bg-amber-50 border-amber-200 text-amber-700"
        }`}>
          <div className="flex items-center gap-2">
            <span>{toast.severity === "CRITICAL" ? "!!" : "!"}</span>
            <span>{toast.message}</span>
            <button onClick={() => setToast(null)} className="ml-auto text-zinc-400 hover:text-zinc-700 cursor-pointer">x</button>
          </div>
        </div>
      )}

      {/* Bell */}
      <div className="relative">
        <button onClick={() => setOpen(!open)} className="relative p-2 rounded-lg glass hover:bg-zinc-100 transition-all cursor-pointer text-zinc-700">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
            <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
          </svg>
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 text-[10px] font-bold text-white flex items-center justify-center">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          )}
        </button>

        {/* Dropdown */}
        {open && (
          <div className="absolute right-0 top-12 w-80 rounded-xl border border-zinc-200 bg-white/95 backdrop-blur-xl shadow-2xl z-50 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200">
              <span className="text-xs font-semibold text-zinc-600 uppercase tracking-widest">Bildirishnomalar</span>
              {unreadCount > 0 && (
                <button onClick={handleMarkAllRead} className="text-[10px] text-indigo-600 hover:text-indigo-700 cursor-pointer">
                  Barchasini o&apos;qish
                </button>
              )}
            </div>
            <div className="max-h-80 overflow-y-auto">
              {notifications.length === 0 ? (
                <p className="px-4 py-6 text-center text-xs text-zinc-500">Bildirishnomalar yo&apos;q</p>
              ) : (
                notifications.slice(0, 20).map((n) => (
                  <div
                    key={n.id}
                    onClick={() => !n.read && handleMarkRead(n.id)}
                    className={`px-4 py-3 border-b border-zinc-100 text-xs cursor-pointer hover:bg-zinc-50 transition-colors ${
                      !n.read ? "bg-indigo-50/50" : ""
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${
                        n.severity === "CRITICAL" ? "bg-red-500" : "bg-amber-500"
                      }`} />
                      <div>
                        <p className="text-zinc-700">{n.message}</p>
                        <p className="text-zinc-500 mt-1">{new Date(n.created_at).toLocaleString("uz")}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
