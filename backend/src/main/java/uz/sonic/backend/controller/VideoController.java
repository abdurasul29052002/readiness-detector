package uz.sonic.backend.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import uz.sonic.backend.dto.VideoFrameResultDto;
import uz.sonic.backend.dto.VideoJobDetailResponse;
import uz.sonic.backend.dto.VideoJobResponse;
import uz.sonic.backend.entity.VideoJob;
import uz.sonic.backend.service.VideoProcessingService;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/video")
@RequiredArgsConstructor
@Slf4j
public class VideoController {

    private final VideoProcessingService videoProcessingService;

    @PostMapping("/upload")
    public ResponseEntity<VideoJobResponse> uploadVideo(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "confidence", defaultValue = "0.5") double confidence,
            @RequestParam(value = "frameInterval", defaultValue = "30") int frameInterval
    ) throws IOException {
        log.info("Video upload: {}, size={}MB", file.getOriginalFilename(), file.getSize() / 1024 / 1024);

        VideoJob job = videoProcessingService.createJob(file.getOriginalFilename(), confidence, frameInterval);
        videoProcessingService.processVideoAsync(job.getId(), file.getBytes(), confidence, frameInterval);

        return ResponseEntity.ok(VideoJobResponse.from(job));
    }

    @GetMapping("/jobs")
    public ResponseEntity<List<VideoJobResponse>> listJobs() {
        return ResponseEntity.ok(
                videoProcessingService.getAllJobs().stream().map(VideoJobResponse::from).toList()
        );
    }

    @GetMapping("/jobs/{id}")
    public ResponseEntity<VideoJobDetailResponse> getJobDetail(@PathVariable Long id) {
        VideoJob job = videoProcessingService.getJob(id);
        List<VideoFrameResultDto> frames = job.getFrameResults().stream()
                .map(VideoFrameResultDto::from).toList();
        return ResponseEntity.ok(new VideoJobDetailResponse(VideoJobResponse.from(job), frames));
    }
}
