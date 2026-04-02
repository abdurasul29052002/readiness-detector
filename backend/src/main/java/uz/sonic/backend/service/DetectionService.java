package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import uz.sonic.backend.dto.BatchClassifyResponse;
import uz.sonic.backend.dto.DetectionResponse;

import java.util.List;

@Service
@RequiredArgsConstructor
public class DetectionService {

    private final RestTemplate restTemplate;

    @Value("${ai-server.url}")
    private String aiServerUrl;

    public DetectionResponse detect(byte[] imageBytes, double confidence) {
        // Multipart request tayyorlash
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentType(MediaType.IMAGE_JPEG);

        ByteArrayResource resource = new ByteArrayResource(imageBytes) {
            @Override
            public String getFilename() {
                return "frame.jpg";
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new HttpEntity<>(resource, fileHeaders));

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        // AI predicter ga yuborish
        String url = aiServerUrl + "/predict?confidence=" + confidence;
        ResponseEntity<DetectionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, request, DetectionResponse.class
        );

        return response.getBody();
    }

    public BatchClassifyResponse classifyBatch(List<byte[]> crops, double confidence) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        for (int i = 0; i < crops.size(); i++) {
            HttpHeaders fileHeaders = new HttpHeaders();
            fileHeaders.setContentType(MediaType.IMAGE_JPEG);

            final int index = i;
            ByteArrayResource resource = new ByteArrayResource(crops.get(i)) {
                @Override
                public String getFilename() {
                    return "crop_" + index + ".jpg";
                }
            };

            body.add("files", new HttpEntity<>(resource, fileHeaders));
        }

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);

        String url = aiServerUrl + "/classify/batch?confidence=" + confidence;
        ResponseEntity<BatchClassifyResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, request, BatchClassifyResponse.class
        );

        return response.getBody();
    }

    public boolean isAiServerHealthy() {
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                    aiServerUrl + "/health", String.class
            );
            return response.getStatusCode().is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }
}
