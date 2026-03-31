package uz.sonic.backend.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import uz.sonic.backend.dto.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DetectionServiceTest {

    @Mock
    private RestTemplate restTemplate;

    @InjectMocks
    private DetectionService detectionService;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(detectionService, "aiServerUrl", "http://localhost:8000");
    }

    @Test
    void detect_shouldReturnDetectionResponse() {
        DetectionResponse expected = new DetectionResponse(
                List.of(new DetectionResult(0, "attentive", 0.9, "attentive",
                        new BoundingBox(10, 20, 100, 200))),
                new DetectionSummary(1, 1, 0, 100.0, 0.0)
        );

        when(restTemplate.exchange(anyString(), eq(HttpMethod.POST), any(), eq(DetectionResponse.class)))
                .thenReturn(new ResponseEntity<>(expected, HttpStatus.OK));

        DetectionResponse result = detectionService.detect(new byte[]{1, 2, 3}, 0.5);

        assertNotNull(result);
        assertEquals(1, result.summary().total());
        assertEquals(1, result.summary().attentive());
    }

    @Test
    void detect_shouldPropagateExceptionWhenAiServerDown() {
        when(restTemplate.exchange(anyString(), eq(HttpMethod.POST), any(), eq(DetectionResponse.class)))
                .thenThrow(new ResourceAccessException("Connection refused"));

        assertThrows(ResourceAccessException.class,
                () -> detectionService.detect(new byte[]{1, 2, 3}, 0.5));
    }

    @Test
    void isAiServerHealthy_shouldReturnTrueWhenServerResponds() {
        when(restTemplate.getForEntity(anyString(), eq(String.class)))
                .thenReturn(new ResponseEntity<>("OK", HttpStatus.OK));

        assertTrue(detectionService.isAiServerHealthy());
    }

    @Test
    void isAiServerHealthy_shouldReturnFalseWhenServerDown() {
        when(restTemplate.getForEntity(anyString(), eq(String.class)))
                .thenThrow(new ResourceAccessException("Connection refused"));

        assertFalse(detectionService.isAiServerHealthy());
    }
}
