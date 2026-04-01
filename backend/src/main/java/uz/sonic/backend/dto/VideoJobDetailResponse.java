package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record VideoJobDetailResponse(
        VideoJobResponse job,
        @JsonProperty("frame_results") List<VideoFrameResultDto> frameResults
) {}
