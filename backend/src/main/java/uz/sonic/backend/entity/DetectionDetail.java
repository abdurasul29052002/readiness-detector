package uz.sonic.backend.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "detection_details")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DetectionDetail {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "session_id")
    private DetectionSession session;

    private int classId;
    private String className;
    private double confidence;
    private String groupName;

    private double bboxX1;
    private double bboxY1;
    private double bboxX2;
    private double bboxY2;
}
