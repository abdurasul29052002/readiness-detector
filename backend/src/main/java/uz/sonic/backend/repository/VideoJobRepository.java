package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import uz.sonic.backend.entity.VideoJob;

import java.util.List;

public interface VideoJobRepository extends JpaRepository<VideoJob, Long> {
    List<VideoJob> findAllByOrderByCreatedAtDesc();
}
