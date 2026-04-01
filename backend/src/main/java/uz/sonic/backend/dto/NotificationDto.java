package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import uz.sonic.backend.entity.Notification;

import java.time.LocalDateTime;

public record NotificationDto(
        Long id,
        String message,
        String severity,
        @JsonProperty("distracted_percent") double distractedPercent,
        double threshold,
        boolean read,
        @JsonProperty("created_at") LocalDateTime createdAt,
        @JsonProperty("camera_name") String cameraName
) {
    public static NotificationDto from(Notification n) {
        return new NotificationDto(
                n.getId(),
                n.getMessage(),
                n.getSeverity(),
                n.getDistractedPercent(),
                n.getThreshold(),
                n.isRead(),
                n.getCreatedAt(),
                n.getCamera() != null ? n.getCamera().getName() : null
        );
    }
}
