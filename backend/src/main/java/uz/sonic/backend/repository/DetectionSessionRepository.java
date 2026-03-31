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
                CAST(ds.timestamp AS LocalDate),
                AVG(ds.attentivePercent),
                AVG(ds.distractedPercent),
                COUNT(ds)
            )
            FROM DetectionSession ds
            WHERE ds.timestamp BETWEEN :start AND :end
            GROUP BY CAST(ds.timestamp AS LocalDate)
            ORDER BY CAST(ds.timestamp AS LocalDate)
            """)
    List<DailyStatistic> findDailyStatistics(
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end
    );
}
