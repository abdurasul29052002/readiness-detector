package uz.sonic.backend.dto;

import java.util.List;

public record BatchClassifyResponse(
        List<ClassificationResult> results,
        DetectionSummary summary
) {}
