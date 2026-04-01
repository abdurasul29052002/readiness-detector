package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record NotificationConfigDto(
        @JsonProperty("distracted_threshold") double distractedThreshold,
        boolean enabled,
        @JsonProperty("sound_enabled") boolean soundEnabled
) {}
