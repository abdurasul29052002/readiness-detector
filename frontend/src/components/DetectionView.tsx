"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DetectionResponse, DetectionResult } from "@/types/detection";
import { detect } from "@/lib/api";
import StatsPanel from "./StatsPanel";

const BOX_COLORS: Record<string, string> = {
  attentive: "#10b981",
  distracted: "#ef4444",
};

export default function DetectionView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedSrc, setUploadedSrc] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);

  const drawDetections = useCallback((detections: DetectionResult[], width: number, height: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    for (const det of detections) {
      const color = BOX_COLORS[det.group] || "#6366f1";
      const { x1, y1, x2, y2 } = det.bbox;
      const w = x2 - x1;
      const h = y2 - y1;

      ctx.shadowColor = color;
      ctx.shadowBlur = 8;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(x1, y1, w, h, 4);
      ctx.stroke();

      ctx.shadowBlur = 0;

      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "600 12px Inter, sans-serif";
      const textW = ctx.measureText(label).width + 12;
      const labelH = 22;
      const labelY = y1 - labelH - 2;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x1, labelY < 0 ? y1 : labelY, textW, labelH, [4, 4, labelY < 0 ? 0 : 4, labelY < 0 ? 0 : 4]);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 6, (labelY < 0 ? y1 + 15 : labelY + 15));
    }
  }, []);

  const captureAndDetect = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    const offscreen = document.createElement("canvas");
    offscreen.width = video.videoWidth;
    offscreen.height = video.videoHeight;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const base64 = offscreen.toDataURL("image/jpeg", 0.85).split(",")[1];

    try {
      setDetecting(true);
      const res = await detect(base64, confidence);
      setResult(res);
      setError(null);
      drawDetections(res.detections, video.videoWidth, video.videoHeight);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Server bilan aloqa xatosi");
    } finally {
      setDetecting(false);
    }
  }, [confidence, drawDetections]);

  const startCamera = async () => {
    try {
      setUploadedSrc(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "environment" },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setIsRunning(true);
      setError(null);
      intervalRef.current = setInterval(captureAndDetect, 1000);
    } catch {
      setError("Kamerani ochib bo'lmadi");
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    setIsRunning(false);
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    stopCamera();
    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result as string;
      const base64 = dataUrl.split(",")[1];
      setUploadedSrc(dataUrl);

      try {
        setDetecting(true);
        const res = await detect(base64, confidence);
        setResult(res);
        setError(null);
        const img = new Image();
        img.onload = () => drawDetections(res.detections, img.width, img.height);
        img.src = dataUrl;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Xatolik");
      } finally {
        setDetecting(false);
      }
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    return () => stopCamera();
  }, []);

  useEffect(() => {
    if (isRunning && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(captureAndDetect, 1000);
    }
  }, [captureAndDetect, isRunning]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-6" style={{ animation: "fade-in-up 0.5s ease" }}>
      {/* Left - Video Panel */}
      <div className="glass rounded-2xl p-5">
        {/* Controls */}
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          {!isRunning ? (
            <button onClick={startCamera}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 text-white text-sm font-semibold shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:-translate-y-0.5 transition-all cursor-pointer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="m15 10-4 4V6l-4 4"/><path d="M14 2.2A10 10 0 0 1 22 12c0 5.5-4.5 10-10 10S2 17.5 2 12a10 10 0 0 1 4-8"/>
              </svg>
              Kamerani yoqish
            </button>
          ) : (
            <button onClick={stopCamera}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-red-500 to-red-600 text-white text-sm font-semibold shadow-lg shadow-red-500/25 hover:shadow-red-500/40 hover:-translate-y-0.5 transition-all cursor-pointer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
              </svg>
              To&#39;xtatish
            </button>
          )}

          <label className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl glass glass-hover text-sm font-semibold cursor-pointer transition-all">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            Rasm yuklash
            <input type="file" accept="image/*" onChange={handleUpload} className="hidden" />
          </label>

          {/* Confidence */}
          <div className="flex items-center gap-3 ml-auto px-4 py-2 rounded-xl glass">
            <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-widest">Ishonch</span>
            <input type="range" min={0.1} max={0.9} step={0.05}
              value={confidence} onChange={e => setConfidence(parseFloat(e.target.value))}
              className="w-24 h-1 rounded-full bg-white/10 appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-500 [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(99,102,241,0.5)] [&::-webkit-slider-thumb]:cursor-pointer" />
            <span className="text-sm font-bold text-indigo-400 min-w-[36px] text-right">
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Video area */}
        <div className="relative rounded-xl overflow-hidden bg-black aspect-video border border-white/5">
          {isRunning && (
            <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-500/90 backdrop-blur-sm text-[10px] font-bold text-white tracking-widest uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-white" style={{ animation: "pulse-live 1.5s infinite" }} />
              LIVE
            </div>
          )}

          {detecting && (
            <div className="absolute top-3 right-3 z-10 px-3 py-1 rounded-full bg-black/60 backdrop-blur-sm text-[10px] font-semibold text-zinc-400 flex items-center gap-2">
              <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round"/>
              </svg>
              Aniqlanmoqda...
            </div>
          )}

          <video ref={videoRef} autoPlay playsInline muted
            className={`w-full h-full object-contain ${isRunning ? 'block' : 'hidden'}`} />

          {uploadedSrc && !isRunning && (
            // eslint-disable-next-line @next/next/no-img-element
            <img ref={imgRef} src={uploadedSrc} alt="Uploaded"
              className="w-full h-full object-contain" />
          )}

          <canvas ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none" />

          {!isRunning && !uploadedSrc && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <div className="w-20 h-20 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-600">
                  <path d="m15 10-4 4V6l-4 4" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="12" cy="12" r="10"/>
                </svg>
              </div>
              <p className="text-sm font-medium text-zinc-600">Kamerani yoqing yoki rasm yuklang</p>
              <p className="text-xs text-zinc-700">Real-time o&#39;quvchi xatti-harakatini aniqlash</p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm font-medium flex items-center gap-2"
            style={{ animation: "fade-in-up 0.3s ease" }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            {error}
          </div>
        )}
      </div>

      {/* Right - Stats */}
      <StatsPanel data={result} />
    </div>
  );
}
