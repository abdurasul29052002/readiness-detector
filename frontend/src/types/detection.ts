export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionResult {
  class_id: number;
  class_name: string;
  confidence: number;
  group: string;
  bbox: BoundingBox;
}

export interface DetectionSummary {
  total: number;
  attentive: number;
  distracted: number;
  attentive_percent: number;
  distracted_percent: number;
}

export interface DetectionResponse {
  detections: DetectionResult[];
  summary: DetectionSummary;
}

// Classification (individual student)
export interface ClassificationResult {
  class_id: number;
  class_name: string;
  confidence: number;
  group: "attentive" | "distracted";
}

export interface BatchClassifyResponse {
  results: ClassificationResult[];
  summary: DetectionSummary;
}

// Face detection (frontend-side)
export interface FaceBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Camera
export interface Camera {
  id: number;
  name: string;
  description: string;
  active: boolean;
  created_at: string;
}

// WebSocket
export interface WebSocketResponseData {
  type: "detection" | "error";
  data: DetectionResponse | null;
  message: string | null;
  timestamp: string;
}

// Notification
export interface NotificationItem {
  id: number;
  message: string;
  severity: "WARNING" | "CRITICAL";
  distracted_percent: number;
  threshold: number;
  read: boolean;
  created_at: string;
  camera_name: string | null;
}

export interface NotificationConfig {
  distracted_threshold: number;
  enabled: boolean;
  sound_enabled: boolean;
}

// Video
export interface VideoJob {
  id: number;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  original_filename: string;
  total_frames: number;
  processed_frames: number;
  frame_interval: number;
  overall_attentive_percent: number;
  overall_distracted_percent: number;
  created_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface VideoFrameResult {
  frame_number: number;
  timestamp_seconds: number;
  total_detected: number;
  attentive_count: number;
  distracted_count: number;
  attentive_percent: number;
  distracted_percent: number;
}

export interface VideoJobDetail {
  job: VideoJob;
  frame_results: VideoFrameResult[];
}

// Statistics
export interface DailyStatistic {
  date: string;
  avg_attentive_percent: number;
  avg_distracted_percent: number;
  session_count: number;
}

export interface StatisticsResponse {
  period_start: string;
  period_end: string;
  overall_avg_attentive: number;
  overall_avg_distracted: number;
  total_sessions: number;
  daily_breakdown: DailyStatistic[];
}

// Model
export interface ModelInfo {
  version: string;
  filename: string;
  loaded: boolean;
  training_date: string | null;
  accuracy: number | null;
  description: string | null;
  task: string | null;
  activation: string | null;
}

export interface ModelListResponse {
  active_version: string;
  models: ModelInfo[];
}
