package uz.sonic.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record ModelInfo(
        String version,
        String filename,
        boolean loaded,
        @JsonProperty("training_date") String trainingDate,
        Double accuracy,
        String description,
        String task
) {}
