package uz.sonic.backend.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import uz.sonic.backend.dto.ModelListResponse;
import uz.sonic.backend.service.ModelService;

import java.util.Map;

@RestController
@RequestMapping("/api/models")
@RequiredArgsConstructor
@Slf4j
public class ModelController {

    private final ModelService modelService;

    @GetMapping
    public ResponseEntity<ModelListResponse> listModels() {
        return ResponseEntity.ok(modelService.listModels());
    }

    @PostMapping("/switch")
    public ResponseEntity<Map<String, Object>> switchModel(@RequestParam String version) {
        log.info("Switching model to version: {}", version);
        return ResponseEntity.ok(modelService.switchModel(version));
    }
}
