package uz.sonic.backend.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.web.bind.annotation.*;
import uz.sonic.backend.dto.LoginRequest;
import uz.sonic.backend.dto.LoginResponse;
import uz.sonic.backend.dto.UserDto;
import uz.sonic.backend.entity.User;
import uz.sonic.backend.repository.UserRepository;
import uz.sonic.backend.service.JwtService;
import uz.sonic.backend.service.UserService;

import java.security.Principal;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
@Slf4j
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final JwtService jwtService;
    private final UserService userService;
    private final UserRepository userRepository;

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(@Valid @RequestBody LoginRequest request) {
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.username(), request.password())
        );

        User user = userRepository.findByUsername(request.username())
                .orElseThrow();

        String token = jwtService.generateToken(user);
        userService.updateLastLogin(user.getUsername());

        log.info("User logged in: {}", user.getUsername());

        return ResponseEntity.ok(new LoginResponse(
                token,
                "Bearer",
                jwtService.getExpirationMs() / 1000,
                UserDto.from(user)
        ));
    }

    @GetMapping("/me")
    public ResponseEntity<UserDto> getCurrentUser(Principal principal) {
        return ResponseEntity.ok(userService.getUserByUsername(principal.getName()));
    }
}
