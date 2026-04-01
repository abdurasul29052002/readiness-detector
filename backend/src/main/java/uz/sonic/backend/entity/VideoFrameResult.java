package uz.sonic.backend.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "video_frame_results")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class VideoFrameResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "video_job_id")
    private VideoJob videoJob;

    private int frameNumber;
    private double timestampSeconds;
    private int totalDetected;
    private int attentiveCount;
    private int distractedCount;
    private double attentivePercent;
    private double distractedPercent;
}
