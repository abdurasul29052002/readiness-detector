package uz.sonic.backend.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import uz.sonic.backend.dto.NotificationConfigDto;
import uz.sonic.backend.dto.NotificationDto;
import uz.sonic.backend.service.NotificationService;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/notifications")
@RequiredArgsConstructor
@Slf4j
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping
    public ResponseEntity<List<NotificationDto>> getNotifications(
            @RequestParam(value = "unreadOnly", defaultValue = "false") boolean unreadOnly) {
        return ResponseEntity.ok(notificationService.getNotifications(unreadOnly));
    }

    @GetMapping("/count")
    public ResponseEntity<Map<String, Long>> getUnreadCount() {
        return ResponseEntity.ok(Map.of("count", notificationService.getUnreadCount()));
    }

    @PatchMapping("/{id}/read")
    public ResponseEntity<Void> markAsRead(@PathVariable Long id) {
        notificationService.markAsRead(id);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/read-all")
    public ResponseEntity<Void> markAllAsRead() {
        notificationService.markAllAsRead();
        return ResponseEntity.ok().build();
    }

    @GetMapping("/config")
    public ResponseEntity<NotificationConfigDto> getConfig() {
        return ResponseEntity.ok(notificationService.getConfig());
    }

    @PutMapping("/config")
    public ResponseEntity<NotificationConfigDto> updateConfig(@RequestBody NotificationConfigDto dto) {
        return ResponseEntity.ok(notificationService.updateConfig(dto));
    }
}
