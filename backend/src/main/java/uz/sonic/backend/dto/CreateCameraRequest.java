package uz.sonic.backend.dto;

import jakarta.validation.constraints.NotBlank;

public record CreateCameraRequest(
        @NotBlank String name,
        String description
) {}
