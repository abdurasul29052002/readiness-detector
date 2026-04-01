package uz.sonic.backend.integration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.*;
import uz.sonic.backend.dto.StatisticsResponse;
import uz.sonic.backend.entity.DetectionSession;

import java.time.LocalDate;
import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;

class StatisticsIntegrationTest extends IntegrationTestBase {

    @BeforeEach
    void setup() {
        sessionRepository.deleteAll();

        LocalDateTime day1 = LocalDate.now().atTime(10, 0);
        LocalDateTime day2 = LocalDate.now().minusDays(1).atTime(10, 0);

        // Day1: 2 sessions
        sessionRepository.save(DetectionSession.builder()
                .timestamp(day1)
                .confidenceThreshold(0.5)
                .totalDetected(10).attentiveCount(7).distractedCount(3)
                .attentivePercent(70.0).distractedPercent(30.0).source("TEST").build());

        sessionRepository.save(DetectionSession.builder()
                .timestamp(day1.plusHours(1))
                .confidenceThreshold(0.5)
                .totalDetected(8).attentiveCount(4).distractedCount(4)
                .attentivePercent(50.0).distractedPercent(50.0).source("TEST").build());

        // Day2: 1 session
        sessionRepository.save(DetectionSession.builder()
                .timestamp(day2)
                .confidenceThreshold(0.5)
                .totalDetected(12).attentiveCount(10).distractedCount(2)
                .attentivePercent(83.3).distractedPercent(16.7).source("TEST").build());
    }

    @Test
    void dailyStatistics_shouldReturn200ForAdmin() {
        HttpEntity<Void> entity = new HttpEntity<>(authHeaders());
        String url = baseUrl() + "/api/statistics/daily?date=" + LocalDate.now();

        ResponseEntity<StatisticsResponse> response = restTemplate.exchange(
                url, HttpMethod.GET, entity, StatisticsResponse.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(2, response.getBody().totalSessions());
    }

    @Test
    void statisticsEndpoints_shouldReturn401WithoutAuth() {
        String url = baseUrl() + "/api/statistics/daily?date=" + LocalDate.now();
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
        assertEquals(HttpStatus.UNAUTHORIZED, response.getStatusCode());
    }
}
