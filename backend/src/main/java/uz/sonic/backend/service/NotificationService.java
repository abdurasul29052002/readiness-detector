package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import uz.sonic.backend.dto.NotificationConfigDto;
import uz.sonic.backend.dto.NotificationDto;
import uz.sonic.backend.entity.Camera;
import uz.sonic.backend.entity.DetectionSession;
import uz.sonic.backend.entity.Notification;
import uz.sonic.backend.entity.NotificationConfig;
import uz.sonic.backend.repository.NotificationConfigRepository;
import uz.sonic.backend.repository.NotificationRepository;
import uz.sonic.backend.websocket.NotificationWebSocketHandler;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private final NotificationRepository notificationRepository;
    private final NotificationConfigRepository configRepository;
    private final NotificationWebSocketHandler webSocketHandler;

    public void checkAndNotify(DetectionSession session) {
        NotificationConfig config = getOrCreateConfig();
        if (!config.isEnabled()) return;
        if (session.getDistractedPercent() < config.getDistractedThreshold()) return;

        String severity = session.getDistractedPercent() >= 80.0 ? "CRITICAL" : "WARNING";
        Camera camera = session.getCamera();
        String cameraInfo = camera != null ? " (kamera: " + camera.getName() + ")" : "";
        String message = String.format("Ogohlantirish: %.0f%% o'quvchilar chalg'igan%s",
                session.getDistractedPercent(), cameraInfo);

        Notification notification = Notification.builder()
                .message(message)
                .severity(severity)
                .distractedPercent(session.getDistractedPercent())
                .threshold(config.getDistractedThreshold())
                .session(session)
                .camera(camera)
                .createdAt(LocalDateTime.now())
                .build();

        Notification saved = notificationRepository.save(notification);
        log.info("Notification created: {} - {}", severity, message);

        webSocketHandler.broadcast(NotificationDto.from(saved));
    }

    public List<NotificationDto> getNotifications(boolean unreadOnly) {
        List<Notification> list = unreadOnly
                ? notificationRepository.findByReadFalseOrderByCreatedAtDesc()
                : notificationRepository.findAllByOrderByCreatedAtDesc();
        return list.stream().map(NotificationDto::from).toList();
    }

    public void markAsRead(Long id) {
        notificationRepository.findById(id).ifPresent(n -> {
            n.setRead(true);
            notificationRepository.save(n);
        });
    }

    public void markAllAsRead() {
        notificationRepository.findByReadFalseOrderByCreatedAtDesc().forEach(n -> {
            n.setRead(true);
            notificationRepository.save(n);
        });
    }

    public long getUnreadCount() {
        return notificationRepository.countByReadFalse();
    }

    public NotificationConfigDto getConfig() {
        NotificationConfig config = getOrCreateConfig();
        return new NotificationConfigDto(config.getDistractedThreshold(), config.isEnabled(), config.isSoundEnabled());
    }

    public NotificationConfigDto updateConfig(NotificationConfigDto dto) {
        NotificationConfig config = getOrCreateConfig();
        config.setDistractedThreshold(dto.distractedThreshold());
        config.setEnabled(dto.enabled());
        config.setSoundEnabled(dto.soundEnabled());
        config.setUpdatedAt(LocalDateTime.now());
        configRepository.save(config);
        return dto;
    }

    private NotificationConfig getOrCreateConfig() {
        return configRepository.findFirstByOrderByIdAsc()
                .orElseGet(() -> configRepository.save(NotificationConfig.builder()
                        .distractedThreshold(60.0)
                        .enabled(true)
                        .soundEnabled(true)
                        .updatedAt(LocalDateTime.now())
                        .build()));
    }
}
