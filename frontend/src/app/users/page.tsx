"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import type { UserProfile, CreateUserRequest } from "@/types/auth";
import { getUsers, createUser, deactivateUser } from "@/lib/api";

export default function UsersPage() {
  const { isAuthenticated, isAdmin, loading: authLoading } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<CreateUserRequest>({ username: "", password: "", full_name: "", role: "TEACHER" });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && (!isAuthenticated || !isAdmin)) {
      router.push("/login");
    }
  }, [authLoading, isAuthenticated, isAdmin, router]);

  useEffect(() => {
    if (isAdmin) {
      getUsers().then(setUsers).catch(() => {});
    }
  }, [isAdmin]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await createUser(form);
      setShowForm(false);
      setForm({ username: "", password: "", full_name: "", role: "TEACHER" });
      setUsers(await getUsers());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Xatolik");
    }
  };

  const handleDeactivate = async (id: number) => {
    await deactivateUser(id);
    setUsers(await getUsers());
  };

  if (authLoading || !isAdmin) return null;

  return (
    <div style={{ animation: "fade-in-up 0.5s ease" }}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gradient">Foydalanuvchilar</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:-translate-y-0.5 transition-all cursor-pointer"
        >
          {showForm ? "Bekor qilish" : "Foydalanuvchi qo'shish"}
        </button>
      </div>

      {showForm && (
        <div className="glass rounded-2xl p-6 mb-6">
          <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Login</label>
              <input type="text" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Parol</label>
              <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required minLength={6}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">To&apos;liq ism</label>
              <input type="text" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} required
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Rol</label>
              <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-sm focus:outline-none focus:border-indigo-500/50 transition-all">
                <option value="TEACHER" className="bg-zinc-900">O&apos;qituvchi</option>
                <option value="ADMIN" className="bg-zinc-900">Admin</option>
              </select>
            </div>
            <div className="md:col-span-2">
              {error && <p className="text-red-400 text-sm mb-2">{error}</p>}
              <button type="submit" className="px-6 py-2.5 rounded-xl bg-emerald-500 text-white text-sm font-semibold hover:bg-emerald-600 transition-all cursor-pointer">
                Saqlash
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="glass rounded-2xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left px-5 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-widest">Login</th>
              <th className="text-left px-5 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-widest">Ism</th>
              <th className="text-left px-5 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-widest">Rol</th>
              <th className="text-left px-5 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-widest">Holat</th>
              <th className="text-right px-5 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-widest">Amallar</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
                <td className="px-5 py-3 font-medium">{u.username}</td>
                <td className="px-5 py-3 text-zinc-400">{u.full_name}</td>
                <td className="px-5 py-3">
                  <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                    u.role === "ADMIN" ? "bg-indigo-500/20 text-indigo-400" : "bg-emerald-500/20 text-emerald-400"
                  }`}>
                    {u.role === "ADMIN" ? "Admin" : "O'qituvchi"}
                  </span>
                </td>
                <td className="px-5 py-3">
                  <span className={`inline-flex items-center gap-1.5 text-xs ${u.active ? "text-emerald-400" : "text-red-400"}`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${u.active ? "bg-emerald-400" : "bg-red-400"}`} />
                    {u.active ? "Faol" : "Nofaol"}
                  </span>
                </td>
                <td className="px-5 py-3 text-right">
                  {u.active && u.role !== "ADMIN" && (
                    <button onClick={() => handleDeactivate(u.id)}
                      className="text-xs text-red-400 hover:text-red-300 transition-colors cursor-pointer">
                      O&apos;chirish
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
