"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import type { Camera } from "@/types/detection";
import { getCameras, createCamera, deleteCamera, toggleCamera } from "@/lib/api";
import DetectionView from "@/components/DetectionView";

export default function CamerasPage() {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const router = useRouter();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) router.push("/login");
  }, [authLoading, isAuthenticated, router]);

  useEffect(() => {
    if (isAuthenticated) getCameras().then(setCameras).catch(() => {});
  }, [isAuthenticated]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    await createCamera(name, description);
    setShowForm(false);
    setName("");
    setDescription("");
    setCameras(await getCameras());
  };

  const handleDelete = async (id: number) => {
    await deleteCamera(id);
    setCameras(await getCameras());
  };

  const handleToggle = async (id: number) => {
    await toggleCamera(id);
    setCameras(await getCameras());
  };

  if (authLoading || !isAuthenticated) return null;

  const activeCameras = cameras.filter((c) => c.active);

  return (
    <div style={{ animation: "fade-in-up 0.5s ease" }}>
      {/* Camera Management */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gradient">Kameralar</h2>
        <button onClick={() => setShowForm(!showForm)}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:-translate-y-0.5 transition-all cursor-pointer">
          {showForm ? "Bekor qilish" : "Kamera qo'shish"}
        </button>
      </div>

      {showForm && (
        <div className="glass rounded-2xl p-6 mb-6">
          <form onSubmit={handleCreate} className="flex gap-4 items-end flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Nomi</label>
              <input type="text" value={name} onChange={(e) => setName(e.target.value)} required
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all"
                placeholder="Sinf xonasi 101" />
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Tavsif</label>
              <input type="text" value={description} onChange={(e) => setDescription(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all"
                placeholder="1-qavat, asosiy bino" />
            </div>
            <button type="submit" className="px-6 py-2.5 rounded-xl bg-emerald-500 text-white text-sm font-semibold hover:bg-emerald-600 transition-all cursor-pointer">
              Saqlash
            </button>
          </form>
        </div>
      )}

      {/* Camera List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {cameras.map((cam) => (
          <div key={cam.id} className={`glass rounded-xl p-4 border transition-all cursor-pointer ${
            selectedCamera === cam.id ? "border-indigo-500/50 glow-indigo" : "border-transparent hover:border-white/10"
          }`} onClick={() => setSelectedCamera(cam.id === selectedCamera ? null : cam.id)}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-sm">{cam.name}</h3>
              <span className={`inline-flex items-center gap-1.5 text-xs ${cam.active ? "text-emerald-400" : "text-zinc-600"}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${cam.active ? "bg-emerald-400" : "bg-zinc-600"}`} />
                {cam.active ? "Faol" : "Nofaol"}
              </span>
            </div>
            {cam.description && <p className="text-xs text-zinc-500 mb-3">{cam.description}</p>}
            <div className="flex gap-2">
              <button onClick={(e) => { e.stopPropagation(); handleToggle(cam.id); }}
                className="text-xs px-3 py-1 rounded-lg glass hover:bg-white/5 transition-all cursor-pointer">
                {cam.active ? "O'chirish" : "Yoqish"}
              </button>
              <button onClick={(e) => { e.stopPropagation(); handleDelete(cam.id); }}
                className="text-xs px-3 py-1 rounded-lg text-red-400 hover:bg-red-500/10 transition-all cursor-pointer">
                O&apos;chirish
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Multi-Camera View */}
      {activeCameras.length > 0 && (
        <>
          <h3 className="text-lg font-bold mb-4">Jonli kuzatuv</h3>
          <div className={`grid gap-6 ${activeCameras.length === 1 ? "grid-cols-1" : "grid-cols-1 lg:grid-cols-2"}`}>
            {activeCameras.map((cam) => (
              <div key={cam.id}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-sm font-semibold">{cam.name}</span>
                </div>
                <DetectionView cameraId={cam.id} compact />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
