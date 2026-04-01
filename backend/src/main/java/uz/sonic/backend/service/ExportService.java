package uz.sonic.backend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import uz.sonic.backend.entity.DetectionDetail;
import uz.sonic.backend.entity.DetectionSession;
import uz.sonic.backend.repository.DetectionSessionRepository;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class ExportService {

    private final DetectionSessionRepository sessionRepository;

    public byte[] exportCsv(LocalDate start, LocalDate end) {
        List<DetectionSession> sessions = sessionRepository.findByTimestampBetween(
                start.atStartOfDay(), end.atTime(LocalTime.MAX));

        StringBuilder sb = new StringBuilder();
        sb.append('\uFEFF'); // BOM for Excel UTF-8
        sb.append("Sana,Vaqt,Jami,Diqqatli,Chalg'igan,Diqqatli %,Chalg'igan %,Manba\n");

        DateTimeFormatter dateFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern("HH:mm:ss");

        for (DetectionSession s : sessions) {
            sb.append(String.format("%s,%s,%d,%d,%d,%.1f,%.1f,%s\n",
                    s.getTimestamp().format(dateFmt),
                    s.getTimestamp().format(timeFmt),
                    s.getTotalDetected(),
                    s.getAttentiveCount(),
                    s.getDistractedCount(),
                    s.getAttentivePercent(),
                    s.getDistractedPercent(),
                    s.getSource()));
        }

        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] exportDetailedCsv(LocalDate start, LocalDate end) {
        List<DetectionSession> sessions = sessionRepository.findByTimestampBetweenWithDetails(
                start.atStartOfDay(), end.atTime(LocalTime.MAX));

        StringBuilder sb = new StringBuilder();
        sb.append('\uFEFF');
        sb.append("Sessiya ID,Vaqt,Sinf ID,Sinf nomi,Ishonch,Guruh,X1,Y1,X2,Y2\n");

        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        for (DetectionSession s : sessions) {
            for (DetectionDetail d : s.getDetails()) {
                sb.append(String.format("%d,%s,%d,%s,%.3f,%s,%.1f,%.1f,%.1f,%.1f\n",
                        s.getId(),
                        s.getTimestamp().format(fmt),
                        d.getClassId(),
                        d.getClassName(),
                        d.getConfidence(),
                        d.getGroupName(),
                        d.getBboxX1(), d.getBboxY1(), d.getBboxX2(), d.getBboxY2()));
            }
        }

        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] exportPdf(LocalDate start, LocalDate end) {
        List<DetectionSession> sessions = sessionRepository.findByTimestampBetween(
                start.atStartOfDay(), end.atTime(LocalTime.MAX));

        long totalSessions = sessions.size();
        double avgAttentive = sessions.stream().mapToDouble(DetectionSession::getAttentivePercent).average().orElse(0);
        double avgDistracted = sessions.stream().mapToDouble(DetectionSession::getDistractedPercent).average().orElse(0);

        // Generate simple HTML-based PDF content
        String html = String.format("""
                <html><head><style>
                body { font-family: Arial, sans-serif; padding: 40px; }
                h1 { color: #333; border-bottom: 2px solid #6366f1; padding-bottom: 10px; }
                .summary { display: flex; gap: 20px; margin: 20px 0; }
                .card { padding: 15px; border: 1px solid #ddd; border-radius: 8px; text-align: center; flex: 1; }
                .card h3 { margin: 0; color: #666; font-size: 12px; text-transform: uppercase; }
                .card p { margin: 5px 0 0; font-size: 24px; font-weight: bold; }
                .green { color: #10b981; }
                .red { color: #ef4444; }
                table { width: 100%%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 8px 12px; border: 1px solid #e5e7eb; text-align: left; }
                th { background: #f9fafb; font-size: 11px; text-transform: uppercase; color: #666; }
                </style></head><body>
                <h1>O'quvchi xatti-harakati hisoboti</h1>
                <p>Davr: %s — %s</p>
                <div class="summary">
                <div class="card"><h3>Sessiyalar</h3><p>%d</p></div>
                <div class="card"><h3>O'rtacha diqqatli</h3><p class="green">%.1f%%</p></div>
                <div class="card"><h3>O'rtacha chalg'igan</h3><p class="red">%.1f%%</p></div>
                </div>
                <table><tr><th>Vaqt</th><th>Jami</th><th>Diqqatli</th><th>Chalg'igan</th><th>Diqqatli %%</th><th>Chalg'igan %%</th></tr>
                """, start, end, totalSessions, avgAttentive, avgDistracted);

        StringBuilder sb = new StringBuilder(html);
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
        for (DetectionSession s : sessions) {
            sb.append(String.format("<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%.1f%%</td><td>%.1f%%</td></tr>",
                    s.getTimestamp().format(fmt), s.getTotalDetected(), s.getAttentiveCount(),
                    s.getDistractedCount(), s.getAttentivePercent(), s.getDistractedPercent()));
        }
        sb.append("</table></body></html>");

        // Return HTML as PDF-like content (real PDF generation would require openhtmltopdf)
        // For now, return as HTML that can be printed/saved as PDF by browser
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }
}
