package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import uz.sonic.backend.entity.VideoJob;

import java.time.LocalDateTime;

public record VideoJobResponse(
        Long id,
        String status,
        @JsonProperty("original_filename") String originalFilename,
        @JsonProperty("total_frames") int totalFrames,
        @JsonProperty("processed_frames") int processedFrames,
        @JsonProperty("frame_interval") int frameInterval,
        @JsonProperty("overall_attentive_percent") double overallAttentivePercent,
        @JsonProperty("overall_distracted_percent") double overallDistractedPercent,
        @JsonProperty("created_at") LocalDateTime createdAt,
        @JsonProperty("completed_at") LocalDateTime completedAt,
        @JsonProperty("error_message") String errorMessage
) {
    public static VideoJobResponse from(VideoJob job) {
        return new VideoJobResponse(
                job.getId(), job.getStatus(), job.getOriginalFilename(),
                job.getTotalFrames(), job.getProcessedFrames(), job.getFrameInterval(),
                job.getOverallAttentivePercent(), job.getOverallDistractedPercent(),
                job.getCreatedAt(), job.getCompletedAt(), job.getErrorMessage());
    }
}
