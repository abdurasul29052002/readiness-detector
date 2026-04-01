import { describe, it, expect, vi, beforeEach } from "vitest";
import { detect, healthCheck, login } from "../api";

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe("detect", () => {
  it("should return detection response on success", async () => {
    const mockResponse = {
      detections: [{ class_id: 0, class_name: "hand-raising", confidence: 0.95, group: "attentive", bbox: { x1: 10, y1: 20, x2: 50, y2: 60 } }],
      summary: { total: 1, attentive: 1, distracted: 0, attentive_percent: 100, distracted_percent: 0 },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    // base64 encoded single pixel JPEG
    const base64 = btoa("test-image-data");
    const result = await detect(base64, 0.5);

    expect(result).toEqual(mockResponse);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toContain("/api/detect");
  });

  it("should throw on server error", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });

    await expect(detect(btoa("data"), 0.5)).rejects.toThrow("503");
  });
});

describe("healthCheck", () => {
  it("should return health status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ status: "ok", aiServer: true }),
    });

    const result = await healthCheck();
    expect(result.status).toBe("ok");
    expect(result.aiServer).toBe(true);
  });

  it("should throw on failure", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await expect(healthCheck()).rejects.toThrow();
  });
});

describe("login", () => {
  it("should return token on success", async () => {
    const mockResponse = {
      token: "jwt-token",
      token_type: "Bearer",
      expires_in: 86400,
      user: { id: 1, username: "admin", full_name: "Admin", role: "ADMIN", active: true, created_at: "", last_login_at: null },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await login("admin", "admin123");
    expect(result.token).toBe("jwt-token");
    expect(result.user.username).toBe("admin");
  });

  it("should throw on wrong credentials", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 401 });
    await expect(login("wrong", "wrong")).rejects.toThrow("noto'g'ri");
  });
});
