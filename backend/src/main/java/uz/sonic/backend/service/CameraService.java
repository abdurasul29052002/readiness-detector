package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import uz.sonic.backend.entity.Camera;
import uz.sonic.backend.repository.CameraRepository;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class CameraService {

    private final CameraRepository cameraRepository;

    public Camera createCamera(String name, String description) {
        Camera camera = Camera.builder()
                .name(name)
                .description(description)
                .active(true)
                .createdAt(LocalDateTime.now())
                .build();
        log.info("Camera created: {}", name);
        return cameraRepository.save(camera);
    }

    public Camera updateCamera(Long id, String name, String description) {
        Camera camera = cameraRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Camera not found: " + id));
        camera.setName(name);
        camera.setDescription(description);
        return cameraRepository.save(camera);
    }

    public void deleteCamera(Long id) {
        cameraRepository.deleteById(id);
    }

    public List<Camera> getAllCameras() {
        return cameraRepository.findAll();
    }

    public List<Camera> getActiveCameras() {
        return cameraRepository.findByActiveTrue();
    }

    public Camera getCamera(Long id) {
        return cameraRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Camera not found: " + id));
    }

    public Camera toggleActive(Long id) {
        Camera camera = getCamera(id);
        camera.setActive(!camera.isActive());
        return cameraRepository.save(camera);
    }
}
