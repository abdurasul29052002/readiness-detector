"use client";

import { useEffect, useRef, useState } from "react";
import type { ModelListResponse } from "@/types/detection";
import { listModels, switchModel } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export default function ModelPanel() {
  const { isAuthenticated } = useAuth();
  const [data, setData] = useState<ModelListResponse | null>(null);
  const [open, setOpen] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listModels().then(setData).catch(() => {});
  }, []);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open]);

  const handleSwitch = async (version: string) => {
    if (!isAuthenticated || switching) return;
    setSwitching(version);
    try {
      await switchModel(version);
      const updated = await listModels();
      setData(updated);
      setOpen(false);
    } catch (e) {
      console.error("Model switch failed:", e);
    } finally {
      setSwitching(null);
    }
  };

  if (!data) return null;

  return (
    <div className="relative" ref={panelRef}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs font-semibold text-zinc-400 hover:text-zinc-300 transition-all cursor-pointer"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        </svg>
        {data.active_version}
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-56 rounded-xl border border-white/10 bg-zinc-900 shadow-2xl z-[100] overflow-hidden">
          <div className="px-3 py-2 border-b border-white/10">
            <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest">Model tanlang</span>
          </div>
          <div className="p-1.5 flex flex-col gap-1">
            {data.models.map((m) => {
              const isActive = m.version === data.active_version;
              const isLoading = switching === m.version;
              return (
                <button
                  key={m.version}
                  onClick={() => !isActive && isAuthenticated && handleSwitch(m.version)}
                  disabled={!!switching || isActive || !isAuthenticated}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-all
                    ${isActive
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 cursor-default"
                      : isAuthenticated
                        ? "text-zinc-300 hover:bg-white/10 hover:text-white cursor-pointer"
                        : "text-zinc-500 cursor-not-allowed"
                    }
                    ${isLoading ? "opacity-50" : ""}
                    ${!!switching && !isActive ? "disabled:opacity-40 disabled:cursor-wait" : ""}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span>{m.version}</span>
                    {isActive && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/30 text-indigo-300">Faol</span>
                    )}
                    {isLoading && (
                      <span className="text-[9px] text-zinc-400">Yuklanmoqda...</span>
                    )}
                  </div>
                  {m.task && <p className="text-[10px] text-zinc-500 mt-0.5">{m.task}</p>}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
