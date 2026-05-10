"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import type { StatisticsResponse } from "@/types/detection";
import { fetchDailyStatistics, fetchWeeklyStatistics, fetchRangeStatistics, exportCsv, exportPdf } from "@/lib/api";

type Mode = "daily" | "weekly" | "range";

function today() {
  return new Date().toISOString().split("T")[0];
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function StatisticsDashboard() {
  const { isAuthenticated } = useAuth();
  const [mode, setMode] = useState<Mode>("daily");
  const [date, setDate] = useState(today());
  const [weekStart, setWeekStart] = useState(today());
  const [rangeStart, setRangeStart] = useState(today());
  const [rangeEnd, setRangeEnd] = useState(today());
  const [data, setData] = useState<StatisticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      let result: StatisticsResponse;
      if (mode === "daily") result = await fetchDailyStatistics(date);
      else if (mode === "weekly") result = await fetchWeeklyStatistics(weekStart);
      else result = await fetchRangeStatistics(rangeStart, rangeEnd);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Xatolik");
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) return null;

  const breakdown = data?.daily_breakdown || [];

  return (
    <div style={{ animation: "fade-in-up 0.5s ease" }}>
      <h2 className="text-xl font-bold text-gradient mb-6">Statistika</h2>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {([["daily", "Kunlik"], ["weekly", "Haftalik"], ["range", "Oraliq"]] as const).map(([m, label]) => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
              mode === m ? "bg-indigo-500/15 text-indigo-700 border border-indigo-500/30" : "glass text-zinc-600 hover:text-zinc-900"
            }`}>
            {label}
          </button>
        ))}
      </div>

      {/* Date Pickers + Fetch */}
      <div className="flex gap-4 items-end mb-6 flex-wrap">
        {mode === "daily" && (
          <div>
            <label className="block text-xs font-semibold text-zinc-600 uppercase tracking-widest mb-2">Sana</label>
            <input type="date" value={date} onChange={(e) => setDate(e.target.value)}
              className="px-4 py-2.5 rounded-xl bg-white border border-zinc-300 text-zinc-900 text-sm focus:outline-none focus:border-indigo-500 transition-all" />
          </div>
        )}
        {mode === "weekly" && (
          <div>
            <label className="block text-xs font-semibold text-zinc-600 uppercase tracking-widest mb-2">Hafta boshlanishi</label>
            <input type="date" value={weekStart} onChange={(e) => setWeekStart(e.target.value)}
              className="px-4 py-2.5 rounded-xl bg-white border border-zinc-300 text-zinc-900 text-sm focus:outline-none focus:border-indigo-500 transition-all" />
          </div>
        )}
        {mode === "range" && (
          <>
            <div>
              <label className="block text-xs font-semibold text-zinc-600 uppercase tracking-widest mb-2">Boshlanish</label>
              <input type="date" value={rangeStart} onChange={(e) => setRangeStart(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-white border border-zinc-300 text-zinc-900 text-sm focus:outline-none focus:border-indigo-500 transition-all" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-600 uppercase tracking-widest mb-2">Tugash</label>
              <input type="date" value={rangeEnd} onChange={(e) => setRangeEnd(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-white border border-zinc-300 text-zinc-900 text-sm focus:outline-none focus:border-indigo-500 transition-all" />
            </div>
          </>
        )}
        <button onClick={fetchData} disabled={loading}
          className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 transition-all disabled:opacity-50 cursor-pointer">
          {loading ? "Yuklanmoqda..." : "Ko'rsatish"}
        </button>
      </div>

      {error && (
        <div className="mb-6 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">{error}</div>
      )}

      {data && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="glass rounded-xl p-4 text-center">
              <p className="text-xs text-zinc-600 uppercase tracking-widest mb-1">Sessiyalar</p>
              <p className="text-2xl font-bold text-zinc-900">{data.total_sessions}</p>
            </div>
            <div className="glass rounded-xl p-4 text-center">
              <p className="text-xs text-zinc-600 uppercase tracking-widest mb-1">Diqqatli</p>
              <p className="text-2xl font-bold text-emerald-600">{data.overall_avg_attentive.toFixed(1)}%</p>
            </div>
            <div className="glass rounded-xl p-4 text-center">
              <p className="text-xs text-zinc-600 uppercase tracking-widest mb-1">Chalg&apos;igan</p>
              <p className="text-2xl font-bold text-red-600">{data.overall_avg_distracted.toFixed(1)}%</p>
            </div>
            <div className="glass rounded-xl p-4 text-center">
              <p className="text-xs text-zinc-600 uppercase tracking-widest mb-1">Davr</p>
              <p className="text-sm font-semibold text-zinc-700">{data.period_start} — {data.period_end}</p>
            </div>
          </div>

          {/* SVG Bar Chart */}
          {breakdown.length > 0 && (
            <div className="glass rounded-2xl p-5 mb-6">
              <h3 className="text-sm font-semibold text-zinc-700 mb-4">Kunlik taqsimot</h3>
              <div className="overflow-x-auto">
                <svg viewBox={`0 0 ${Math.max(breakdown.length * 80, 400)} 220`} className="w-full min-w-[400px]" preserveAspectRatio="xMinYMid meet">
                  {/* Y-axis labels */}
                  {[0, 25, 50, 75, 100].map((v) => (
                    <g key={v}>
                      <text x="30" y={200 - v * 1.8 + 4} fill="#71717a" fontSize="10" textAnchor="end">{v}%</text>
                      <line x1="35" y1={200 - v * 1.8} x2={breakdown.length * 80 + 40} y2={200 - v * 1.8} stroke="#e4e4e7" strokeWidth="0.5" />
                    </g>
                  ))}
                  {/* Bars */}
                  {breakdown.map((d, i) => {
                    const x = 50 + i * 80;
                    const attH = d.avg_attentive_percent * 1.8;
                    const distH = d.avg_distracted_percent * 1.8;
                    return (
                      <g key={d.date}>
                        <rect x={x} y={200 - attH} width="25" height={attH} rx="3" fill="#10b981" opacity="0.85" />
                        <rect x={x + 30} y={200 - distH} width="25" height={distH} rx="3" fill="#ef4444" opacity="0.85" />
                        <text x={x + 27} y="215" fill="#52525b" fontSize="9" textAnchor="middle">
                          {d.date.slice(5)}
                        </text>
                        <text x={x + 27} y="212" fill="#71717a" fontSize="8" textAnchor="middle" dy="15">
                          ({d.session_count})
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
              <div className="flex gap-4 mt-3 text-xs text-zinc-600">
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-500" /> Diqqatli</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-red-500" /> Chalg&apos;igan</span>
              </div>
            </div>
          )}

          {/* Export Buttons */}
          <div className="glass rounded-2xl p-5 mb-6">
            <h3 className="text-sm font-semibold text-zinc-700 mb-3">Eksport</h3>
            <div className="flex gap-3 flex-wrap">
              <button onClick={async () => { const blob = await exportCsv(data.period_start, data.period_end); downloadBlob(blob, `statistika_${data.period_start}_${data.period_end}.csv`); }}
                className="px-4 py-2 rounded-xl glass glass-hover text-sm font-semibold text-emerald-700 cursor-pointer transition-all">
                CSV yuklash
              </button>
              <button onClick={async () => { const blob = await exportPdf(data.period_start, data.period_end); downloadBlob(blob, `hisobot_${data.period_start}_${data.period_end}.pdf`); }}
                className="px-4 py-2 rounded-xl glass glass-hover text-sm font-semibold text-indigo-700 cursor-pointer transition-all">
                PDF hisobot
              </button>
            </div>
          </div>

          {/* Daily Breakdown Table */}
          {breakdown.length > 0 && (
            <div className="glass rounded-2xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200">
                    <th className="text-left px-5 py-3 text-xs font-semibold text-zinc-600 uppercase tracking-widest">Sana</th>
                    <th className="text-right px-5 py-3 text-xs font-semibold text-zinc-600 uppercase tracking-widest">Sessiyalar</th>
                    <th className="text-right px-5 py-3 text-xs font-semibold text-zinc-600 uppercase tracking-widest">Diqqatli</th>
                    <th className="text-right px-5 py-3 text-xs font-semibold text-zinc-600 uppercase tracking-widest">Chalg&apos;igan</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.map((d) => (
                    <tr key={d.date} className="border-b border-zinc-100 hover:bg-zinc-50 transition-colors">
                      <td className="px-5 py-3 font-medium text-zinc-800">{d.date}</td>
                      <td className="px-5 py-3 text-right text-zinc-600">{d.session_count}</td>
                      <td className="px-5 py-3 text-right text-emerald-600">{d.avg_attentive_percent.toFixed(1)}%</td>
                      <td className="px-5 py-3 text-right text-red-600">{d.avg_distracted_percent.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
