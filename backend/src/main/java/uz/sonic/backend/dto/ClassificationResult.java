package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record ClassificationResult(
        @JsonProperty("class_id") int classId,
        @JsonProperty("class_name") String className,
        double confidence,
        String group
) {}
