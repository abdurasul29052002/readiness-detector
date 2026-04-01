package uz.sonic.backend.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import uz.sonic.backend.dto.CreateUserRequest;
import uz.sonic.backend.dto.UpdateUserRequest;
import uz.sonic.backend.dto.UserDto;
import uz.sonic.backend.service.UserService;

import java.util.List;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@Slf4j
public class UserController {

    private final UserService userService;

    @GetMapping
    public ResponseEntity<List<UserDto>> listUsers() {
        return ResponseEntity.ok(userService.getAllUsers());
    }

    @PostMapping
    public ResponseEntity<UserDto> createUser(@Valid @RequestBody CreateUserRequest request) {
        log.info("Creating user: {}", request.username());
        return ResponseEntity.ok(UserDto.from(userService.createUser(request)));
    }

    @PutMapping("/{id}")
    public ResponseEntity<UserDto> updateUser(@PathVariable Long id,
                                              @Valid @RequestBody UpdateUserRequest request) {
        log.info("Updating user: {}", id);
        return ResponseEntity.ok(UserDto.from(userService.updateUser(id, request)));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deactivateUser(@PathVariable Long id) {
        log.info("Deactivating user: {}", id);
        userService.deactivateUser(id);
        return ResponseEntity.noContent().build();
    }
}
