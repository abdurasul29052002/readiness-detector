package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record DetectionSummary(
        int total,
        int attentive,
        int distracted,
        @JsonProperty("attentive_percent") double attentivePercent,
        @JsonProperty("distracted_percent") double distractedPercent
) {}
