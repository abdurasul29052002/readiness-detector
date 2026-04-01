package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import uz.sonic.backend.entity.VideoFrameResult;

public record VideoFrameResultDto(
        @JsonProperty("frame_number") int frameNumber,
        @JsonProperty("timestamp_seconds") double timestampSeconds,
        @JsonProperty("total_detected") int totalDetected,
        @JsonProperty("attentive_count") int attentiveCount,
        @JsonProperty("distracted_count") int distractedCount,
        @JsonProperty("attentive_percent") double attentivePercent,
        @JsonProperty("distracted_percent") double distractedPercent
) {
    public static VideoFrameResultDto from(VideoFrameResult r) {
        return new VideoFrameResultDto(
                r.getFrameNumber(), r.getTimestampSeconds(),
                r.getTotalDetected(), r.getAttentiveCount(), r.getDistractedCount(),
                r.getAttentivePercent(), r.getDistractedPercent());
    }
}
