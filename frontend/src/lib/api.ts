import { DetectionResponse } from "@/types/detection";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

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

export async function healthCheck(): Promise<{ status: string; aiServer: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
