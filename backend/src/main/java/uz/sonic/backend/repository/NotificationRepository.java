package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import uz.sonic.backend.entity.Notification;

import java.util.List;

public interface NotificationRepository extends JpaRepository<Notification, Long> {
    List<Notification> findByReadFalseOrderByCreatedAtDesc();
    List<Notification> findAllByOrderByCreatedAtDesc();
    long countByReadFalse();
}
