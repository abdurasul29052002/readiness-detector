package uz.sonic.backend.websocket;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.AbstractWebSocketHandler;
import uz.sonic.backend.dto.DetectionResponse;
import uz.sonic.backend.dto.WebSocketRequest;
import uz.sonic.backend.dto.WebSocketResponse;
import uz.sonic.backend.service.DetectionPersistenceService;
import uz.sonic.backend.service.DetectionService;

import java.util.Base64;

@Component
@RequiredArgsConstructor
@Slf4j
public class DetectionWebSocketHandler extends AbstractWebSocketHandler {

    private final DetectionService detectionService;
    private final DetectionPersistenceService persistenceService;
    private final ObjectMapper objectMapper;

    @Value("${websocket.default-confidence:0.5}")
    private double defaultConfidence;

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        try {
            WebSocketRequest request = objectMapper.readValue(message.getPayload(), WebSocketRequest.class);
            double confidence = request.confidence() > 0 ? request.confidence() : defaultConfidence;
            byte[] imageBytes = Base64.getDecoder().decode(request.image());

            DetectionResponse response = detectionService.detect(imageBytes, confidence);
            persistenceService.saveDetection(response, confidence, "WEBSOCKET");

            String json = objectMapper.writeValueAsString(WebSocketResponse.success(response));
            session.sendMessage(new TextMessage(json));
        } catch (Exception e) {
            log.error("WebSocket text message processing error: {}", e.getMessage());
            String errorJson = objectMapper.writeValueAsString(WebSocketResponse.error(e.getMessage()));
            session.sendMessage(new TextMessage(errorJson));
        }
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) throws Exception {
        try {
            byte[] imageBytes = message.getPayload().array();

            DetectionResponse response = detectionService.detect(imageBytes, defaultConfidence);
            persistenceService.saveDetection(response, defaultConfidence, "WEBSOCKET");

            String json = objectMapper.writeValueAsString(WebSocketResponse.success(response));
            session.sendMessage(new TextMessage(json));
        } catch (Exception e) {
            log.error("WebSocket binary message processing error: {}", e.getMessage());
            String errorJson = objectMapper.writeValueAsString(WebSocketResponse.error(e.getMessage()));
            session.sendMessage(new TextMessage(errorJson));
        }
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        log.info("WebSocket connected: {}", session.getId());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        log.info("WebSocket disconnected: {}, status: {}", session.getId(), status);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {
        log.error("WebSocket transport error [{}]: {}", session.getId(), exception.getMessage());
    }
}
