package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import uz.sonic.backend.dto.ModelListResponse;

import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class ModelService {

    private final RestTemplate restTemplate;

    @Value("${ai-server.url}")
    private String aiServerUrl;

    public ModelListResponse listModels() {
        ResponseEntity<ModelListResponse> response = restTemplate.getForEntity(
                aiServerUrl + "/models", ModelListResponse.class);
        return response.getBody();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> switchModel(String version) {
        ResponseEntity<Map> response = restTemplate.postForEntity(
                aiServerUrl + "/models/switch?version=" + version, null, Map.class);
        return response.getBody();
    }
}
