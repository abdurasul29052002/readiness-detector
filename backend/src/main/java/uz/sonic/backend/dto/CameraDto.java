package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import uz.sonic.backend.entity.Camera;

import java.time.LocalDateTime;

public record CameraDto(
        Long id,
        String name,
        String description,
        boolean active,
        @JsonProperty("created_at") LocalDateTime createdAt
) {
    public static CameraDto from(Camera camera) {
        return new CameraDto(
                camera.getId(),
                camera.getName(),
                camera.getDescription(),
                camera.isActive(),
                camera.getCreatedAt()
        );
    }
}
