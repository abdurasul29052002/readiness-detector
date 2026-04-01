package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import uz.sonic.backend.entity.NotificationConfig;

import java.util.Optional;

public interface NotificationConfigRepository extends JpaRepository<NotificationConfig, Long> {
    Optional<NotificationConfig> findFirstByOrderByIdAsc();
}
