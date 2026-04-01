package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record CreateUserRequest(
        @NotBlank String username,
        @NotBlank @Size(min = 6) String password,
        @NotBlank @JsonProperty("full_name") String fullName,
        @NotBlank String role
) {}
