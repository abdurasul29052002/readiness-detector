package uz.sonic.backend.integration;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.context.ActiveProfiles;
import uz.sonic.backend.repository.DetectionSessionRepository;
import uz.sonic.backend.service.JwtService;
import uz.sonic.backend.entity.User;
import uz.sonic.backend.repository.UserRepository;

import org.springframework.http.HttpHeaders;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
public abstract class IntegrationTestBase {

    @LocalServerPort
    protected int port;

    @Autowired
    protected TestRestTemplate restTemplate;

    @Autowired
    protected DetectionSessionRepository sessionRepository;

    @Autowired
    protected JwtService jwtService;

    @Autowired
    protected UserRepository userRepository;

    protected String baseUrl() {
        return "http://localhost:" + port;
    }

    protected HttpHeaders authHeaders() {
        User admin = userRepository.findByUsername("admin").orElseThrow();
        String token = jwtService.generateToken(admin);
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(token);
        return headers;
    }
}
