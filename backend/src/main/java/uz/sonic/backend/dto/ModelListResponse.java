package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record ModelListResponse(
        @JsonProperty("active_version") String activeVersion,
        List<ModelInfo> models
) {}
