"use client";

import { DetectionResponse } from "@/types/detection";

interface Props {
  data: DetectionResponse | null;
}

const CLASS_CONFIG: Record<string, { label: string; icon: string }> = {
  "hand-raising": { label: "Qo'l ko'tarish", icon: "&#9995;" },
  read:           { label: "O'qish",          icon: "&#128214;" },
  write:          { label: "Yozish",          icon: "&#9997;" },
  discuss:        { label: "Suhbatlashish",   icon: "&#128172;" },
  "bow-head":     { label: "Bosh egish",      icon: "&#128532;" },
  "turn-head":    { label: "Boshni burish",   icon: "&#128064;" },
  standing:       { label: "O'rnidan turish",  icon: "&#128694;" },
};

const ATTENTIVE = new Set(["hand-raising", "read", "write"]);

export default function StatsPanel({ data }: Props) {
  if (!data) {
    return (
      <div className="flex flex-col gap-4" style={{ animation: "fade-in-up 0.5s ease 0.1s both" }}>
        <div className="glass rounded-2xl p-10 text-center">
          <div className="w-16 h-16 rounded-full border-2 border-dashed border-white/8 flex items-center justify-center mx-auto mb-4">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-400">
              <path d="M3 3v18h18" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="m7 17 4-8 4 4 4-6" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <p className="text-sm font-medium text-zinc-400">Natijalar shu yerda ko&#39;rinadi</p>
          <p className="text-xs text-zinc-400 mt-1">Kamera yoqilganda avtomatik yangilanadi</p>
        </div>
      </div>
    );
  }

  const { summary, detections } = data;

  const classCounts: Record<string, number> = {};
  for (const d of detections) {
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  }

  const sortedClasses = Object.entries(classCounts).sort((a, b) => {
    const aAtt = ATTENTIVE.has(a[0]) ? 0 : 1;
    const bAtt = ATTENTIVE.has(b[0]) ? 0 : 1;
    if (aAtt !== bAtt) return aAtt - bAtt;
    return b[1] - a[1];
  });

  const attPct = summary.attentive_percent;
  const disPct = summary.distracted_percent;

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const attArc = summary.total > 0 ? (summary.attentive / summary.total) * circumference : 0;
  const disArc = summary.total > 0 ? (summary.distracted / summary.total) * circumference : 0;

  return (
    <div className="flex flex-col gap-4" style={{ animation: "fade-in-up 0.5s ease 0.1s both" }}>
      {/* Donut Chart Card */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-widest mb-5 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 8.5 15"/>
          </svg>
          Umumiy ko&#39;rinish
        </h3>

        <div className="flex justify-center mb-5">
          <div className="relative w-[140px] h-[140px]">
            <svg width="140" height="140" viewBox="0 0 140 140" className="-rotate-90">
              <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="12" />
              <circle cx="70" cy="70" r={radius} fill="none" stroke="#10b981" strokeWidth="12"
                strokeDasharray={`${attArc} ${circumference}`}
                strokeLinecap="round"
                style={{ transition: "stroke-dasharray 0.8s ease" }} />
              <circle cx="70" cy="70" r={radius} fill="none" stroke="#ef4444" strokeWidth="12"
                strokeDasharray={`${disArc} ${circumference}`}
                strokeDashoffset={-attArc}
                strokeLinecap="round"
                style={{ transition: "stroke-dasharray 0.8s ease, stroke-dashoffset 0.8s ease" }} />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-extrabold text-white" style={{ animation: "count-pop 0.4s ease" }}>
                {summary.total}
              </span>
              <span className="text-[9px] font-semibold text-zinc-400 uppercase tracking-widest mt-0.5">
                o&#39;quvchi
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="text-center p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/10">
            <div className="text-2xl font-extrabold text-indigo-400" style={{ animation: "count-pop 0.4s ease" }}>
              {summary.total}
            </div>
            <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mt-1">Jami</div>
          </div>
          <div className="text-center p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/10">
            <div className="text-2xl font-extrabold text-emerald-400" style={{ animation: "count-pop 0.4s ease" }}>
              {summary.attentive}
            </div>
            <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mt-1">Diqqatli</div>
          </div>
          <div className="text-center p-3 rounded-xl bg-red-500/10 border border-red-500/10">
            <div className="text-2xl font-extrabold text-red-400" style={{ animation: "count-pop 0.4s ease" }}>
              {summary.distracted}
            </div>
            <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mt-1">Chalg&#39;igan</div>
          </div>
        </div>

        {summary.total > 0 && (
          <div className="mt-5">
            <div className="h-2.5 rounded-full bg-white/5 overflow-hidden flex">
              <div className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-l-full transition-all duration-700"
                style={{ width: `${attPct}%` }} />
              <div className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-r-full transition-all duration-700"
                style={{ width: `${disPct}%` }} />
            </div>
            <div className="flex justify-between mt-2">
              <span className="text-emerald-400 text-sm font-bold flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 glow-green-sm" />
                {attPct.toFixed(1)}%
              </span>
              <span className="text-red-400 text-sm font-bold flex items-center gap-1.5">
                {disPct.toFixed(1)}%
                <span className="w-2 h-2 rounded-full bg-red-400 glow-red-sm" />
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Class breakdown */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-[11px] font-semibold text-zinc-400 uppercase tracking-widest mb-4 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M3 12h4l3-9 4 18 3-9h4"/>
          </svg>
          Sinflar bo&#39;yicha
        </h3>

        <div className="flex flex-col gap-2">
          {sortedClasses.map(([name, count], i) => {
            const isAtt = ATTENTIVE.has(name);
            const cfg = CLASS_CONFIG[name];
            return (
              <div key={name}
                className="flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04] hover:border-white/[0.08] transition-all"
                style={{ animation: `fade-in-up 0.3s ease ${i * 0.05}s both` }}>
                <div className="flex items-center gap-3">
                  <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                    isAtt ? 'bg-emerald-400 glow-green-sm' : 'bg-red-400 glow-red-sm'
                  }`} />
                  <span className="text-sm" dangerouslySetInnerHTML={{ __html: cfg?.icon || '' }} />
                  <span className="text-[13px] font-medium text-zinc-300">
                    {cfg?.label || name}
                  </span>
                </div>
                <span className={`text-sm font-bold px-2.5 py-0.5 rounded-lg ${
                  isAtt
                    ? 'text-emerald-400 bg-emerald-500/10'
                    : 'text-red-400 bg-red-500/10'
                }`}>
                  {count}
                </span>
              </div>
            );
          })}

          {sortedClasses.length === 0 && (
            <p className="text-center text-zinc-400 text-sm py-4">Hech narsa aniqlanmadi</p>
          )}
        </div>
      </div>
    </div>
  );
}
