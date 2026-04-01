package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record LoginResponse(
        String token,
        @JsonProperty("token_type") String tokenType,
        @JsonProperty("expires_in") long expiresIn,
        UserDto user
) {}
