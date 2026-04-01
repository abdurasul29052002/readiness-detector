package uz.sonic.backend.integration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import uz.sonic.backend.dto.*;
import uz.sonic.backend.entity.DetectionSession;
import uz.sonic.backend.service.DetectionPersistenceService;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DetectionPersistenceIntegrationTest extends IntegrationTestBase {

    @Autowired
    private DetectionPersistenceService detectionPersistenceService;

    @BeforeEach
    void setup() {
        sessionRepository.deleteAll();
    }

    @Test
    void saveDetection_shouldPersistSessionAndDetails() {
        DetectionResponse response = new DetectionResponse(
                List.of(
                        new DetectionResult(0, "hand-raising", 0.95, "attentive",
                                new BoundingBox(10, 20, 50, 60)),
                        new DetectionResult(3, "discuss", 0.8, "distracted",
                                new BoundingBox(100, 200, 300, 400))
                ),
                new DetectionSummary(2, 1, 1, 50.0, 50.0)
        );

        DetectionSession saved = detectionPersistenceService.saveDetection(response, 0.5, "TEST");

        assertNotNull(saved.getId());
        assertEquals(2, saved.getTotalDetected());
        assertEquals(1, saved.getAttentiveCount());
        assertEquals(1, saved.getDistractedCount());
        assertEquals(2, saved.getDetails().size());

        // Verify persisted to DB
        DetectionSession fromDb = sessionRepository.findById(saved.getId()).orElseThrow();
        assertEquals("TEST", fromDb.getSource());
    }

    @Test
    void savedDetections_shouldBeQueryableByTimestamp() {
        DetectionResponse response = new DetectionResponse(
                List.of(),
                new DetectionSummary(0, 0, 0, 0, 0)
        );

        detectionPersistenceService.saveDetection(response, 0.5, "TEST");
        detectionPersistenceService.saveDetection(response, 0.5, "TEST");

        List<DetectionSession> all = sessionRepository.findAll();
        assertEquals(2, all.size());
    }
}
