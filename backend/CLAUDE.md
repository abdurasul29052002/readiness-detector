# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spring Boot 4.0.5 backend for a student behavior detection system (package: `uz.sonic.backend`). Acts as a gateway between a Vaadin Hilla frontend and an external AI prediction server that classifies student attention levels (attentive vs distracted) from images. Detection results are persisted to a database with statistics API for a single admin user.

Java 25, H2 (dev) / PostgreSQL (prod).

## Build & Run Commands

```bash
# Build (skip tests)
./mvnw package -DskipTests

# Run
./mvnw spring-boot:run

# Run tests
./mvnw test

# Run a single test class
./mvnw test -Dtest=DetectionServiceTest

# Docker (full stack with AI server + PostgreSQL)
docker compose up --build
```

## Architecture

Three entry points into `DetectionService`, all persisting results via `DetectionPersistenceService`:

- **REST Controller** (`controller/DetectionController`) — `POST /api/detect` (multipart file), `GET /api/health`
- **Hilla Endpoint** (`endpoint/DetectionEndpoint`) — Base64 images from Vaadin/React UI. `@AnonymousAllowed`
- **WebSocket** (`websocket/DetectionWebSocketHandler`) — `ws://host/ws/detect`. Accepts binary (raw image bytes) or text (JSON with base64 image). Real-time video frame detection.

`DetectionService` forwards images via `RestTemplate` to AI server (`POST /predict?confidence=...`). AI server URL: `ai-server.url` property.

**Statistics** — `StatisticsController` at `/api/statistics/{daily,weekly,range}` (admin auth required). `StatisticsService` queries `DetectionSessionRepository` for aggregated daily averages.

**Persistence** — `DetectionSession` (summary per frame) → `DetectionDetail` (each detected student). Saved by `DetectionPersistenceService`.

## Key Patterns

- DTOs are Java records with `@JsonProperty` for snake_case mapping
- Entities are JPA classes with Lombok (`@Getter`, `@Setter`, `@Builder`)
- Services use `@RequiredArgsConstructor` for constructor injection
- Spring Boot 4.x: `@WebMvcTest` is in `org.springframework.boot.webmvc.test.autoconfigure` (not the old `boot.test.autoconfigure.web.servlet`)
- `@MockitoBean` replaces `@MockBean` (Spring Boot 4.x)

## Security

- HTTP Basic auth, single admin user (in-memory)
- `/api/statistics/**` — authenticated (admin only)
- `/api/detect`, `/api/health`, `/ws/**`, Swagger UI — public
- Admin credentials: `admin.username` / `admin.password` properties

## Key Configuration

- Server port: `8080`
- H2 console: `/h2-console` (dev only)
- Swagger UI: `/swagger-ui.html`
- AI server: `ai-server.url` (default `http://localhost:8000`)
- Profiles: `docker` (PostgreSQL + AI server), `prod` (PostgreSQL), `test` (H2 in-memory)
