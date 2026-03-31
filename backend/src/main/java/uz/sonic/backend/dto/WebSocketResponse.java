package uz.sonic.backend.dto;

import java.time.LocalDateTime;

public record WebSocketResponse(
        String type,
        Object data,
        String message,
        LocalDateTime timestamp
) {
    public static WebSocketResponse success(Object data) {
        return new WebSocketResponse("detection", data, null, LocalDateTime.now());
    }

    public static WebSocketResponse error(String message) {
        return new WebSocketResponse("error", null, message, LocalDateTime.now());
    }
}
