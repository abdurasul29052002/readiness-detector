package uz.sonic.backend.dto;

import java.util.List;

public record DetectionResponse(
        List<DetectionResult> detections,
        DetectionSummary summary
) {}
