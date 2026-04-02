import type { DetectionResponse, BatchClassifyResponse, Camera, NotificationItem, NotificationConfig, VideoJob, VideoJobDetail, StatisticsResponse, ModelListResponse } from "@/types/detection";
import type { LoginResponse, UserProfile, CreateUserRequest, UpdateUserRequest } from "@/types/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// --- Token management ---

let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
  if (token) {
    localStorage.setItem("auth_token", token);
  } else {
    localStorage.removeItem("auth_token");
  }
}

export function getAuthToken(): string | null {
  if (!authToken && typeof window !== "undefined") {
    authToken = localStorage.getItem("auth_token");
  }
  return authToken;
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// --- Auth ---

export async function login(username: string, password: string): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(res.status === 401 ? "Login yoki parol noto'g'ri" : `Server xatosi: ${res.status}`);
  return res.json();
}

export async function getCurrentUser(): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() });
  if (!res.ok) throw new Error("Autentifikatsiya xatosi");
  return res.json();
}

// --- Users (admin) ---

export async function getUsers(): Promise<UserProfile[]> {
  const res = await fetch(`${API_BASE}/api/users`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function createUser(data: CreateUserRequest): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/api/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function updateUser(id: number, data: UpdateUserRequest): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/api/users/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function deactivateUser(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/users/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
}

// --- Detection ---

export async function detect(
  base64Image: string,
  confidence: number
): Promise<DetectionResponse> {
  const byteString = atob(base64Image);
  const bytes = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i++) {
    bytes[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: "image/jpeg" });

  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  formData.append("confidence", confidence.toString());

  const res = await fetch(`${API_BASE}/api/detect`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Server xatosi: ${res.status}`);
  }

  return res.json();
}

export async function classifyBatch(
  crops: Blob[],
  confidence: number
): Promise<BatchClassifyResponse> {
  const formData = new FormData();
  crops.forEach((blob, i) => formData.append("files", blob, `crop_${i}.jpg`));
  formData.append("confidence", confidence.toString());

  const res = await fetch(`${API_BASE}/api/classify/batch`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Server xatosi: ${res.status}`);
  }

  return res.json();
}

export async function healthCheck(): Promise<{ status: string; aiServer: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

// --- Statistics ---

export async function fetchDailyStatistics(date: string) {
  const res = await fetch(`${API_BASE}/api/statistics/daily?date=${date}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function fetchWeeklyStatistics(weekStart: string) {
  const res = await fetch(`${API_BASE}/api/statistics/weekly?weekStart=${weekStart}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function fetchRangeStatistics(start: string, end: string): Promise<StatisticsResponse> {
  const res = await fetch(`${API_BASE}/api/statistics/range?start=${start}&end=${end}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

// --- Cameras ---

export async function getCameras(): Promise<Camera[]> {
  const res = await fetch(`${API_BASE}/api/cameras`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function createCamera(name: string, description: string): Promise<Camera> {
  const res = await fetch(`${API_BASE}/api/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function deleteCamera(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/cameras/${id}`, { method: "DELETE", headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
}

export async function toggleCamera(id: number): Promise<Camera> {
  const res = await fetch(`${API_BASE}/api/cameras/${id}/toggle`, { method: "PATCH", headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

// --- Notifications ---

export async function getNotifications(unreadOnly = false): Promise<NotificationItem[]> {
  const res = await fetch(`${API_BASE}/api/notifications?unreadOnly=${unreadOnly}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function getUnreadCount(): Promise<number> {
  const res = await fetch(`${API_BASE}/api/notifications/count`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  const data = await res.json();
  return data.count;
}

export async function markNotificationRead(id: number): Promise<void> {
  await fetch(`${API_BASE}/api/notifications/${id}/read`, { method: "PATCH", headers: authHeaders() });
}

export async function markAllNotificationsRead(): Promise<void> {
  await fetch(`${API_BASE}/api/notifications/read-all`, { method: "PATCH", headers: authHeaders() });
}

export async function getNotificationConfig(): Promise<NotificationConfig> {
  const res = await fetch(`${API_BASE}/api/notifications/config`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function updateNotificationConfig(config: NotificationConfig): Promise<NotificationConfig> {
  const res = await fetch(`${API_BASE}/api/notifications/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

// --- Video ---

export async function uploadVideo(file: File, confidence: number, frameInterval: number): Promise<VideoJob> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("confidence", confidence.toString());
  formData.append("frameInterval", frameInterval.toString());
  const res = await fetch(`${API_BASE}/api/video/upload`, { method: "POST", headers: authHeaders(), body: formData });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function getVideoJobs(): Promise<VideoJob[]> {
  const res = await fetch(`${API_BASE}/api/video/jobs`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function getVideoJobDetail(id: number): Promise<VideoJobDetail> {
  const res = await fetch(`${API_BASE}/api/video/jobs/${id}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

// --- Export ---

export async function exportCsv(start: string, end: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/export/csv?start=${start}&end=${end}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.blob();
}

export async function exportDetailedCsv(start: string, end: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/export/csv/detailed?start=${start}&end=${end}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.blob();
}

export async function exportPdf(start: string, end: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/export/pdf?start=${start}&end=${end}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.blob();
}

// --- Models ---

export async function listModels(): Promise<ModelListResponse> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}

export async function switchModel(version: string): Promise<{ status: string; active_version: string }> {
  const res = await fetch(`${API_BASE}/api/models/switch?version=${version}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Server xatosi: ${res.status}`);
  return res.json();
}
