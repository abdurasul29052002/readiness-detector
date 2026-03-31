package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.LocalDate;
import java.util.List;

public record StatisticsResponse(
        @JsonProperty("period_start") LocalDate periodStart,
        @JsonProperty("period_end") LocalDate periodEnd,
        @JsonProperty("overall_avg_attentive") double overallAvgAttentive,
        @JsonProperty("overall_avg_distracted") double overallAvgDistracted,
        @JsonProperty("total_sessions") long totalSessions,
        @JsonProperty("daily_breakdown") List<DailyStatistic> dailyBreakdown
) {
}
