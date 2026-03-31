package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDate;

public record DailyStatistic(
        LocalDate date,
        @JsonProperty("avg_attentive_percent") double avgAttentivePercent,
        @JsonProperty("avg_distracted_percent") double avgDistractedPercent,
        @JsonProperty("session_count") long sessionCount
) {
}
