package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import uz.sonic.backend.entity.Camera;

import java.util.List;
import java.util.Optional;

public interface CameraRepository extends JpaRepository<Camera, Long> {
    Optional<Camera> findByName(String name);
    List<Camera> findByActiveTrue();
}
