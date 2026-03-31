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
