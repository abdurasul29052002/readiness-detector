package uz.sonic.backend.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "video_jobs")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class VideoJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String originalFilename;

    @Builder.Default
    private String status = "PENDING"; // PENDING, PROCESSING, COMPLETED, FAILED

    private int totalFrames;
    private int processedFrames;
    private int frameInterval;
    private double confidenceThreshold;

    private LocalDateTime createdAt;
    private LocalDateTime completedAt;

    private String errorMessage;

    private double overallAttentivePercent;
    private double overallDistractedPercent;

    @OneToMany(mappedBy = "videoJob", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<VideoFrameResult> frameResults = new ArrayList<>();
}
