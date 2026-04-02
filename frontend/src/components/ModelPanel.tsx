"use client";

import { useEffect, useRef, useState } from "react";
import type { ModelListResponse } from "@/types/detection";
import { listModels, switchModel } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export default function ModelPanel() {
  const { isAuthenticated } = useAuth();
  const [data, setData] = useState<ModelListResponse | null>(null);
  const [open, setOpen] = useState(false);
  const [switching, setSwitching] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listModels().then(setData).catch(() => {});
  }, []);

  // Tashqariga bosilganda dropdown yopilsin
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
    if (!isAuthenticated) return;
    setSwitching(true);
    try {
      await switchModel(version);
      setData(await listModels());
      setOpen(false);
    } catch {
    } finally {
      setSwitching(false);
    }
  };

  if (!data) return null;

  const active = data.models.find((m) => m.version === data.active_version);

  return (
    <div className="relative" ref={panelRef}>
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs font-semibold text-zinc-400 hover:text-zinc-300 transition-all cursor-pointer">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        </svg>
        {data.active_version}
      </button>

      {open && (
        <div className="absolute right-0 top-10 w-72 glass rounded-xl border border-white/10 shadow-2xl z-50 overflow-hidden">
          <div className="px-4 py-3 border-b border-white/5">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">Model versiyalari</span>
          </div>
          {data.models.map((m) => (
            <div key={m.version} className="px-4 py-3 border-b border-white/5 hover:bg-white/[0.02] transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                    {m.version}
                    {m.version === data.active_version && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400">Faol</span>
                    )}
                  </p>
                  {m.description && <p className="text-[10px] text-zinc-400">{m.description}</p>}
                  {m.accuracy && <p className="text-[10px] text-zinc-500">Aniqlik: {(m.accuracy * 100).toFixed(1)}%</p>}
                  {m.training_date && <p className="text-[10px] text-zinc-600">{m.training_date}</p>}
                </div>
                {m.version !== data.active_version && isAuthenticated && (
                  <button onClick={() => handleSwitch(m.version)} disabled={switching}
                    className="text-[10px] px-2.5 py-1 rounded-lg bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 transition-all cursor-pointer disabled:opacity-50">
                    Tanlash
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
