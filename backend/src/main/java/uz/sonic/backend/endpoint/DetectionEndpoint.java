package uz.sonic.backend.endpoint;

import com.vaadin.flow.server.auth.AnonymousAllowed;
import com.vaadin.hilla.Endpoint;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import uz.sonic.backend.dto.DetectionResponse;
import uz.sonic.backend.service.DetectionPersistenceService;
import uz.sonic.backend.service.DetectionService;

import java.util.Base64;

@Endpoint
@AnonymousAllowed
@RequiredArgsConstructor
@Slf4j
public class DetectionEndpoint {

    private final DetectionService detectionService;
    private final DetectionPersistenceService detectionPersistenceService;

    public DetectionResponse detect(String base64Image, double confidence) {
        byte[] imageBytes = Base64.getDecoder().decode(base64Image);
        DetectionResponse response = detectionService.detect(imageBytes, confidence);
        detectionPersistenceService.saveDetection(response, confidence, "HILLA");
        log.info("Hilla detection: confidence={}, total={}", confidence, response.summary().total());
        return response;
    }
}
