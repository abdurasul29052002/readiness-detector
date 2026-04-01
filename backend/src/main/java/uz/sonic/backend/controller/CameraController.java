package uz.sonic.backend.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import uz.sonic.backend.dto.CameraDto;
import uz.sonic.backend.dto.CreateCameraRequest;
import uz.sonic.backend.service.CameraService;

import java.util.List;

@RestController
@RequestMapping("/api/cameras")
@RequiredArgsConstructor
@Slf4j
public class CameraController {

    private final CameraService cameraService;

    @GetMapping
    public ResponseEntity<List<CameraDto>> listCameras() {
        return ResponseEntity.ok(cameraService.getAllCameras().stream().map(CameraDto::from).toList());
    }

    @PostMapping
    public ResponseEntity<CameraDto> createCamera(@Valid @RequestBody CreateCameraRequest request) {
        log.info("Creating camera: {}", request.name());
        return ResponseEntity.ok(CameraDto.from(cameraService.createCamera(request.name(), request.description())));
    }

    @PutMapping("/{id}")
    public ResponseEntity<CameraDto> updateCamera(@PathVariable Long id,
                                                   @Valid @RequestBody CreateCameraRequest request) {
        return ResponseEntity.ok(CameraDto.from(cameraService.updateCamera(id, request.name(), request.description())));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteCamera(@PathVariable Long id) {
        cameraService.deleteCamera(id);
        return ResponseEntity.noContent().build();
    }

    @PatchMapping("/{id}/toggle")
    public ResponseEntity<CameraDto> toggleActive(@PathVariable Long id) {
        return ResponseEntity.ok(CameraDto.from(cameraService.toggleActive(id)));
    }
}
