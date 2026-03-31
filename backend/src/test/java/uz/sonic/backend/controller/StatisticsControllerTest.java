package uz.sonic.backend.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import uz.sonic.backend.config.SecurityConfig;
import uz.sonic.backend.dto.DailyStatistic;
import uz.sonic.backend.dto.StatisticsResponse;
import uz.sonic.backend.service.StatisticsService;

import java.time.LocalDate;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.user;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(StatisticsController.class)
@Import(SecurityConfig.class)
class StatisticsControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private StatisticsService statisticsService;

    @Test
    void dailyStatistics_shouldReturn200ForAdmin() throws Exception {
        LocalDate today = LocalDate.now();
        StatisticsResponse response = new StatisticsResponse(
                today, today, 75.0, 25.0, 10,
                List.of(new DailyStatistic(today, 75.0, 25.0, 10))
        );
        when(statisticsService.getDailyStatistics(any(LocalDate.class))).thenReturn(response);

        mockMvc.perform(get("/api/statistics/daily").with(user("admin").roles("ADMIN")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total_sessions").value(10))
                .andExpect(jsonPath("$.overall_avg_attentive").value(75.0));
    }

    @Test
    void dailyStatistics_shouldReturn401ForUnauthenticated() throws Exception {
        mockMvc.perform(get("/api/statistics/daily"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void weeklyStatistics_shouldReturn200() throws Exception {
        LocalDate monday = LocalDate.of(2026, 3, 30);
        StatisticsResponse response = new StatisticsResponse(
                monday, monday.plusDays(6), 70.0, 30.0, 50, List.of()
        );
        when(statisticsService.getWeeklyStatistics(any(LocalDate.class))).thenReturn(response);

        mockMvc.perform(get("/api/statistics/weekly")
                        .param("weekStart", "2026-03-30")
                        .with(user("admin").roles("ADMIN")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total_sessions").value(50));
    }
}
