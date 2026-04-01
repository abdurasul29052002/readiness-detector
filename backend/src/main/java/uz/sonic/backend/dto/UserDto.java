package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import uz.sonic.backend.entity.User;

import java.time.LocalDateTime;

public record UserDto(
        Long id,
        String username,
        @JsonProperty("full_name") String fullName,
        String role,
        boolean active,
        @JsonProperty("created_at") LocalDateTime createdAt,
        @JsonProperty("last_login_at") LocalDateTime lastLoginAt
) {
    public static UserDto from(User user) {
        return new UserDto(
                user.getId(),
                user.getUsername(),
                user.getFullName(),
                user.getRole().name(),
                user.isActive(),
                user.getCreatedAt(),
                user.getLastLoginAt()
        );
    }
}
