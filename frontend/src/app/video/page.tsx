"use client";

import { useEffect, useState, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import type { VideoJob, VideoJobDetail } from "@/types/detection";
import { uploadVideo, getVideoJobs, getVideoJobDetail } from "@/lib/api";

export default function VideoPage() {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const router = useRouter();
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<VideoJobDetail | null>(null);
  const [uploading, setUploading] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [frameInterval, setFrameInterval] = useState(30);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) router.push("/login");
  }, [authLoading, isAuthenticated, router]);

  useEffect(() => {
    if (isAuthenticated) getVideoJobs().then(setJobs).catch(() => {});
  }, [isAuthenticated]);

  // Poll for processing jobs
  useEffect(() => {
    const hasProcessing = jobs.some((j) => j.status === "PENDING" || j.status === "PROCESSING");
    if (hasProcessing) {
      pollingRef.current = setInterval(async () => {
        const updated = await getVideoJobs();
        setJobs(updated);
        if (!updated.some((j) => j.status === "PENDING" || j.status === "PROCESSING")) {
          if (pollingRef.current) clearInterval(pollingRef.current);
        }
      }, 3000);
    }
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [jobs.length]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await uploadVideo(file, confidence, frameInterval);
      setJobs(await getVideoJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Xatolik");
    } finally {
      setUploading(false);
    }
  };

  const handleViewDetail = async (id: number) => {
    const detail = await getVideoJobDetail(id);
    setSelectedJob(detail);
  };

  if (authLoading || !isAuthenticated) return null;

  const results = selectedJob?.frame_results || [];

  return (
    <div style={{ animation: "fade-in-up 0.5s ease" }}>
      <h2 className="text-xl font-bold text-gradient mb-6">Video tahlil</h2>

      {/* Upload Section */}
      <div className="glass rounded-2xl p-6 mb-6">
        <div className="flex gap-4 items-end flex-wrap">
          <div>
            <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Video fayl</label>
            <label className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl glass glass-hover text-sm font-semibold cursor-pointer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              {uploading ? "Yuklanmoqda..." : "Fayl tanlash"}
              <input type="file" accept="video/*" onChange={handleUpload} className="hidden" disabled={uploading} />
            </label>
          </div>
          <div>
            <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Kadr oralig&apos;i</label>
            <select value={frameInterval} onChange={(e) => setFrameInterval(Number(e.target.value))}
              className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all">
              <option value={15} className="bg-zinc-900">Har 15 kadr</option>
              <option value={30} className="bg-zinc-900">Har 30 kadr (~1/sek)</option>
              <option value={60} className="bg-zinc-900">Har 60 kadr (~2/sek)</option>
            </select>
          </div>
          <div className="flex items-center gap-3 px-4 py-2 rounded-xl glass">
            <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-widest">Ishonch</span>
            <input type="range" min={0.1} max={0.9} step={0.05} value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="w-20 h-1 rounded-full bg-white/10 appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-500 [&::-webkit-slider-thumb]:cursor-pointer" />
            <span className="text-sm font-bold text-indigo-400">{(confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
        {error && <p className="mt-3 text-red-400 text-sm">{error}</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-6">
        {/* Jobs List */}
        <div className="glass rounded-2xl overflow-hidden">
          <div className="px-5 py-3 border-b border-white/5">
            <span className="text-xs font-semibold text-zinc-600 uppercase tracking-widest">Ishlar tarixi</span>
          </div>
          {jobs.length === 0 ? (
            <p className="px-5 py-8 text-center text-sm text-zinc-600">Hali video yuklanmagan</p>
          ) : (
            jobs.map((job) => (
              <div key={job.id} onClick={() => job.status === "COMPLETED" && handleViewDetail(job.id)}
                className="px-5 py-3 border-b border-white/5 hover:bg-white/[0.02] transition-colors cursor-pointer">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{job.original_filename}</p>
                    <p className="text-[10px] text-zinc-600">{new Date(job.created_at).toLocaleString("uz")}</p>
                  </div>
                  <span className={`px-2.5 py-1 rounded-full text-[10px] font-semibold ${
                    job.status === "COMPLETED" ? "bg-emerald-500/20 text-emerald-400" :
                    job.status === "FAILED" ? "bg-red-500/20 text-red-400" :
                    "bg-amber-500/20 text-amber-400"
                  }`}>
                    {job.status === "COMPLETED" ? "Tayyor" : job.status === "FAILED" ? "Xato" :
                     job.status === "PROCESSING" ? "Jarayonda..." : "Kutilmoqda"}
                  </span>
                </div>
                {job.status === "COMPLETED" && (
                  <div className="flex gap-4 mt-1 text-[10px] text-zinc-600">
                    <span>Diqqatli: <span className="text-emerald-400">{job.overall_attentive_percent.toFixed(1)}%</span></span>
                    <span>Chalg&apos;igan: <span className="text-red-400">{job.overall_distracted_percent.toFixed(1)}%</span></span>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Timeline Chart */}
        {selectedJob && results.length > 0 && (
          <div className="glass rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-zinc-400 mb-2">{selectedJob.job.original_filename} — Vaqt bo&apos;yicha tahlil</h3>
            <div className="overflow-x-auto">
              <svg viewBox={`0 0 ${Math.max(results.length * 8, 400)} 200`} className="w-full min-w-[400px]" preserveAspectRatio="xMinYMid meet">
                {/* Grid */}
                {[0, 25, 50, 75, 100].map((v) => (
                  <g key={v}>
                    <text x="25" y={180 - v * 1.6 + 4} fill="#52525b" fontSize="9" textAnchor="end">{v}%</text>
                    <line x1="30" y1={180 - v * 1.6} x2={results.length * 8 + 35} y2={180 - v * 1.6} stroke="#27272a" strokeWidth="0.3" />
                  </g>
                ))}
                {/* Lines */}
                <polyline
                  points={results.map((r, i) => `${35 + i * 8},${180 - r.attentive_percent * 1.6}`).join(" ")}
                  fill="none" stroke="#10b981" strokeWidth="1.5" />
                <polyline
                  points={results.map((r, i) => `${35 + i * 8},${180 - r.distracted_percent * 1.6}`).join(" ")}
                  fill="none" stroke="#ef4444" strokeWidth="1.5" />
              </svg>
            </div>
            <div className="flex gap-4 mt-2 text-xs text-zinc-500">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-emerald-500" /> Diqqatli</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-red-500" /> Chalg&apos;igan</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
