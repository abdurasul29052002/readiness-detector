package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import uz.sonic.backend.entity.VideoFrameResult;
import uz.sonic.backend.entity.VideoJob;
import uz.sonic.backend.repository.VideoJobRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class VideoProcessingService {

    private final RestTemplate restTemplate;
    private final VideoJobRepository videoJobRepository;

    @Value("${ai-server.url}")
    private String aiServerUrl;

    public VideoJob createJob(String filename, double confidence, int frameInterval) {
        VideoJob job = VideoJob.builder()
                .originalFilename(filename)
                .status("PENDING")
                .confidenceThreshold(confidence)
                .frameInterval(frameInterval)
                .createdAt(LocalDateTime.now())
                .build();
        return videoJobRepository.save(job);
    }

    @Async("videoProcessingExecutor")
    @SuppressWarnings("unchecked")
    public void processVideoAsync(Long jobId, byte[] videoBytes, double confidence, int frameInterval) {
        VideoJob job = videoJobRepository.findById(jobId).orElse(null);
        if (job == null) return;

        job.setStatus("PROCESSING");
        videoJobRepository.save(job);

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            ByteArrayResource resource = new ByteArrayResource(videoBytes) {
                @Override
                public String getFilename() {
                    return job.getOriginalFilename();
                }
            };
            body.add("file", resource);
            body.add("confidence", String.valueOf(confidence));
            body.add("frame_interval", String.valueOf(frameInterval));

            HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.exchange(
                    aiServerUrl + "/predict/video", HttpMethod.POST, request, Map.class);

            Map<String, Object> result = response.getBody();
            if (result == null) throw new RuntimeException("Empty response from AI server");

            job.setTotalFrames(((Number) result.get("total_frames")).intValue());
            job.setProcessedFrames(((Number) result.get("processed_frames")).intValue());

            Map<String, Object> overall = (Map<String, Object>) result.get("overall_summary");
            job.setOverallAttentivePercent(((Number) overall.get("avg_attentive_percent")).doubleValue());
            job.setOverallDistractedPercent(((Number) overall.get("avg_distracted_percent")).doubleValue());

            List<Map<String, Object>> frames = (List<Map<String, Object>>) result.get("frame_results");
            for (Map<String, Object> fr : frames) {
                Map<String, Object> summary = (Map<String, Object>) fr.get("summary");
                VideoFrameResult vfr = VideoFrameResult.builder()
                        .videoJob(job)
                        .frameNumber(((Number) fr.get("frame_number")).intValue())
                        .timestampSeconds(((Number) fr.get("timestamp_seconds")).doubleValue())
                        .totalDetected(((Number) summary.get("total")).intValue())
                        .attentiveCount(((Number) summary.get("attentive")).intValue())
                        .distractedCount(((Number) summary.get("distracted")).intValue())
                        .attentivePercent(((Number) summary.get("attentive_percent")).doubleValue())
                        .distractedPercent(((Number) summary.get("distracted_percent")).doubleValue())
                        .build();
                job.getFrameResults().add(vfr);
            }

            job.setStatus("COMPLETED");
            job.setCompletedAt(LocalDateTime.now());
            log.info("Video job {} completed: {} frames processed", jobId, job.getProcessedFrames());

        } catch (Exception e) {
            job.setStatus("FAILED");
            job.setErrorMessage(e.getMessage());
            log.error("Video job {} failed: {}", jobId, e.getMessage());
        }

        videoJobRepository.save(job);
    }

    public VideoJob getJob(Long id) {
        return videoJobRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Video job not found: " + id));
    }

    public List<VideoJob> getAllJobs() {
        return videoJobRepository.findAllByOrderByCreatedAtDesc();
    }
}
