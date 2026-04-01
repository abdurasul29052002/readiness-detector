package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record UpdateUserRequest(
        @JsonProperty("full_name") String fullName,
        String password,
        String role
) {}
