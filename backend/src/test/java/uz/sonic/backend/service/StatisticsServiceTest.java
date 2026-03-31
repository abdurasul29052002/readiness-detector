package uz.sonic.backend.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import uz.sonic.backend.dto.DailyStatistic;
import uz.sonic.backend.dto.StatisticsResponse;
import uz.sonic.backend.repository.DetectionSessionRepository;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class StatisticsServiceTest {

    @Mock
    private DetectionSessionRepository sessionRepository;

    @InjectMocks
    private StatisticsService statisticsService;

    @Test
    void getDailyStatistics_shouldReturnCorrectAverages() {
        LocalDate today = LocalDate.now();
        List<DailyStatistic> daily = List.of(
                new DailyStatistic(today, 75.0, 25.0, 10)
        );
        when(sessionRepository.findDailyStatistics(any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(daily);

        StatisticsResponse result = statisticsService.getDailyStatistics(today);

        assertEquals(today, result.periodStart());
        assertEquals(today, result.periodEnd());
        assertEquals(75.0, result.overallAvgAttentive());
        assertEquals(25.0, result.overallAvgDistracted());
        assertEquals(10, result.totalSessions());
    }

    @Test
    void getWeeklyStatistics_shouldSpanSevenDays() {
        LocalDate monday = LocalDate.of(2026, 3, 30);
        when(sessionRepository.findDailyStatistics(any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(Collections.emptyList());

        StatisticsResponse result = statisticsService.getWeeklyStatistics(monday);

        assertEquals(monday, result.periodStart());
        assertEquals(monday.plusDays(6), result.periodEnd());
    }

    @Test
    void getStatistics_shouldHandleEmptyResults() {
        when(sessionRepository.findDailyStatistics(any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(Collections.emptyList());

        StatisticsResponse result = statisticsService.getStatistics(
                LocalDate.of(2026, 3, 1), LocalDate.of(2026, 3, 31));

        assertEquals(0, result.totalSessions());
        assertEquals(0.0, result.overallAvgAttentive());
        assertEquals(0.0, result.overallAvgDistracted());
        assertTrue(result.dailyBreakdown().isEmpty());
    }

    @Test
    void getStatistics_shouldComputeWeightedAverages() {
        LocalDate day1 = LocalDate.of(2026, 3, 30);
        LocalDate day2 = LocalDate.of(2026, 3, 31);
        List<DailyStatistic> daily = List.of(
                new DailyStatistic(day1, 80.0, 20.0, 8),
                new DailyStatistic(day2, 60.0, 40.0, 2)
        );
        when(sessionRepository.findDailyStatistics(any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(daily);

        StatisticsResponse result = statisticsService.getStatistics(day1, day2);

        assertEquals(10, result.totalSessions());
        // Weighted: (80*8 + 60*2) / 10 = 76.0
        assertEquals(76.0, result.overallAvgAttentive());
        // Weighted: (20*8 + 40*2) / 10 = 24.0
        assertEquals(24.0, result.overallAvgDistracted());
    }
}
