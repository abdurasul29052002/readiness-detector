package uz.sonic.backend.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "notification_configs")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class NotificationConfig {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Builder.Default
    private double distractedThreshold = 60.0;

    @Builder.Default
    private boolean enabled = true;

    @Builder.Default
    private boolean soundEnabled = true;

    private LocalDateTime updatedAt;
}
