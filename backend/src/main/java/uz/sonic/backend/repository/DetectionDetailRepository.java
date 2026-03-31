package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import uz.sonic.backend.entity.DetectionDetail;

public interface DetectionDetailRepository extends JpaRepository<DetectionDetail, Long> {
}
