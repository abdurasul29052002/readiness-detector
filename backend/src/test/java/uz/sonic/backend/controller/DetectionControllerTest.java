package uz.sonic.backend.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import uz.sonic.backend.config.SecurityConfig;
import uz.sonic.backend.dto.*;
import uz.sonic.backend.entity.DetectionSession;
import uz.sonic.backend.service.DetectionPersistenceService;
import uz.sonic.backend.service.DetectionService;

import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(DetectionController.class)
@Import(SecurityConfig.class)
class DetectionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private DetectionService detectionService;

    @MockitoBean
    private DetectionPersistenceService detectionPersistenceService;

    @Test
    void detect_shouldReturn200WithValidFile() throws Exception {
        DetectionResponse response = new DetectionResponse(
                List.of(new DetectionResult(0, "attentive", 0.9, "attentive",
                        new BoundingBox(10, 20, 100, 200))),
                new DetectionSummary(1, 1, 0, 100.0, 0.0)
        );
        when(detectionService.detect(any(byte[].class), anyDouble())).thenReturn(response);
        when(detectionPersistenceService.saveDetection(any(), anyDouble(), anyString()))
                .thenReturn(new DetectionSession());

        MockMultipartFile file = new MockMultipartFile(
                "file", "test.jpg", MediaType.IMAGE_JPEG_VALUE, "fake-image".getBytes());

        mockMvc.perform(multipart("/api/detect").file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.summary.total").value(1))
                .andExpect(jsonPath("$.summary.attentive").value(1));
    }

    @Test
    void health_shouldReturnHealthStatus() throws Exception {
        when(detectionService.isAiServerHealthy()).thenReturn(true);

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ok"))
                .andExpect(jsonPath("$.aiServer").value(true));
    }
}
