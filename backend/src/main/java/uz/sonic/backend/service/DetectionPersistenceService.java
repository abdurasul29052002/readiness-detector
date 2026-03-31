package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import uz.sonic.backend.dto.DetectionResponse;
import uz.sonic.backend.dto.DetectionResult;
import uz.sonic.backend.entity.DetectionDetail;
import uz.sonic.backend.entity.DetectionSession;
import uz.sonic.backend.repository.DetectionSessionRepository;

import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
@Slf4j
public class DetectionPersistenceService {

    private final DetectionSessionRepository sessionRepository;

    public DetectionSession saveDetection(DetectionResponse response, double confidence, String source) {
        DetectionSession session = DetectionSession.builder()
                .timestamp(LocalDateTime.now())
                .confidenceThreshold(confidence)
                .totalDetected(response.summary().total())
                .attentiveCount(response.summary().attentive())
                .distractedCount(response.summary().distracted())
                .attentivePercent(response.summary().attentivePercent())
                .distractedPercent(response.summary().distractedPercent())
                .source(source)
                .build();

        for (DetectionResult result : response.detections()) {
            DetectionDetail detail = DetectionDetail.builder()
                    .session(session)
                    .classId(result.classId())
                    .className(result.className())
                    .confidence(result.confidence())
                    .groupName(result.group())
                    .bboxX1(result.bbox().x1())
                    .bboxY1(result.bbox().y1())
                    .bboxX2(result.bbox().x2())
                    .bboxY2(result.bbox().y2())
                    .build();
            session.getDetails().add(detail);
        }

        DetectionSession saved = sessionRepository.save(session);
        log.debug("Detection session saved: id={}, total={}, source={}", saved.getId(), saved.getTotalDetected(), source);
        return saved;
    }
}
