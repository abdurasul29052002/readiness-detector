package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import uz.sonic.backend.dto.DailyStatistic;
import uz.sonic.backend.dto.StatisticsResponse;
import uz.sonic.backend.repository.DetectionSessionRepository;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class StatisticsService {

    private final DetectionSessionRepository sessionRepository;

    public StatisticsResponse getDailyStatistics(LocalDate date) {
        return getStatistics(date, date);
    }

    public StatisticsResponse getWeeklyStatistics(LocalDate weekStart) {
        LocalDate start = weekStart.with(DayOfWeek.MONDAY);
        LocalDate end = start.plusDays(6);
        return getStatistics(start, end);
    }

    public StatisticsResponse getStatistics(LocalDate start, LocalDate end) {
        LocalDateTime startDateTime = start.atStartOfDay();
        LocalDateTime endDateTime = end.plusDays(1).atStartOfDay();

        List<DailyStatistic> daily = sessionRepository.findDailyStatistics(startDateTime, endDateTime);

        long totalSessions = daily.stream().mapToLong(DailyStatistic::sessionCount).sum();

        double overallAttentive = 0;
        double overallDistracted = 0;
        if (totalSessions > 0) {
            overallAttentive = daily.stream()
                    .mapToDouble(d -> d.avgAttentivePercent() * d.sessionCount())
                    .sum() / totalSessions;
            overallDistracted = daily.stream()
                    .mapToDouble(d -> d.avgDistractedPercent() * d.sessionCount())
                    .sum() / totalSessions;
        }

        return new StatisticsResponse(
                start, end,
                Math.round(overallAttentive * 100.0) / 100.0,
                Math.round(overallDistracted * 100.0) / 100.0,
                totalSessions,
                daily
        );
    }
}
