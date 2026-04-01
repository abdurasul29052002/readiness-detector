package uz.sonic.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import uz.sonic.backend.dto.DailyStatistic;
import uz.sonic.backend.entity.DetectionSession;

import java.time.LocalDateTime;
import java.util.List;

public interface DetectionSessionRepository extends JpaRepository<DetectionSession, Long> {

    List<DetectionSession> findByTimestampBetween(LocalDateTime start, LocalDateTime end);

    @Query("""
            SELECT new uz.sonic.backend.dto.DailyStatistic(
                cast(ds.timestamp as LocalDate),
                AVG(ds.attentivePercent),
                AVG(ds.distractedPercent),
                COUNT(ds)
            )
            FROM DetectionSession ds
            WHERE ds.timestamp BETWEEN :start AND :end
            GROUP BY cast(ds.timestamp as LocalDate)
            ORDER BY cast(ds.timestamp as LocalDate)
            """)
    List<DailyStatistic> findDailyStatistics(
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end
    );

    @Query("SELECT ds FROM DetectionSession ds LEFT JOIN FETCH ds.details WHERE ds.timestamp BETWEEN :start AND :end ORDER BY ds.timestamp")
    List<DetectionSession> findByTimestampBetweenWithDetails(
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end
    );
}
