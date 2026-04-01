package uz.sonic.backend.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import uz.sonic.backend.dto.DetectionResponse;
import uz.sonic.backend.entity.Camera;
import uz.sonic.backend.service.CameraService;
import uz.sonic.backend.service.DetectionPersistenceService;
import uz.sonic.backend.service.DetectionService;

import java.io.IOException;
import java.util.Map;

/**
 * REST API — tashqi integratsiya yoki test uchun.
 * Asosiy UI Hilla endpoint orqali ishlaydi.
 */
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@Slf4j
public class DetectionController {

    private final DetectionService detectionService;
    private final DetectionPersistenceService detectionPersistenceService;
    private final CameraService cameraService;

    @PostMapping("/detect")
    public ResponseEntity<DetectionResponse> detect(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "confidence", defaultValue = "0.5") double confidence,
            @RequestParam(value = "cameraId", required = false) Long cameraId
    ) throws IOException {
        DetectionResponse response = detectionService.detect(file.getBytes(), confidence);
        Camera camera = cameraId != null ? cameraService.getCamera(cameraId) : null;
        detectionPersistenceService.saveDetection(response, confidence, "REST", camera);
        log.info("REST detection: confidence={}, total={}", confidence, response.summary().total());
        return ResponseEntity.ok(response);
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "ok",
                "aiServer", detectionService.isAiServerHealthy()
        ));
    }
}
