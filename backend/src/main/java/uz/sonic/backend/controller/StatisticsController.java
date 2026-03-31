package uz.sonic.backend.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import uz.sonic.backend.dto.StatisticsResponse;
import uz.sonic.backend.service.StatisticsService;

import java.time.DayOfWeek;
import java.time.LocalDate;

@RestController
@RequestMapping("/api/statistics")
@RequiredArgsConstructor
@Slf4j
public class StatisticsController {

    private final StatisticsService statisticsService;

    @GetMapping("/daily")
    public ResponseEntity<StatisticsResponse> dailyStatistics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date
    ) {
        LocalDate targetDate = date != null ? date : LocalDate.now();
        log.info("Daily statistics requested for date: {}", targetDate);
        return ResponseEntity.ok(statisticsService.getDailyStatistics(targetDate));
    }

    @GetMapping("/weekly")
    public ResponseEntity<StatisticsResponse> weeklyStatistics(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate weekStart
    ) {
        LocalDate start = weekStart != null ? weekStart : LocalDate.now().with(DayOfWeek.MONDAY);
        log.info("Weekly statistics requested from: {}", start);
        return ResponseEntity.ok(statisticsService.getWeeklyStatistics(start));
    }

    @GetMapping("/range")
    public ResponseEntity<StatisticsResponse> rangeStatistics(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate end
    ) {
        log.info("Range statistics requested: {} to {}", start, end);
        return ResponseEntity.ok(statisticsService.getStatistics(start, end));
    }
}
