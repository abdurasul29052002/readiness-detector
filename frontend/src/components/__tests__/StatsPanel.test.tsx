import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import StatsPanel from "../StatsPanel";

describe("StatsPanel", () => {
  it("renders empty state when data is null", () => {
    render(<StatsPanel data={null} />);
    expect(screen.getByText(/Natijalar shu yerda/i)).toBeInTheDocument();
  });

  it("renders detection data correctly", () => {
    const data = {
      detections: [
        { class_id: 0, class_name: "hand-raising", confidence: 0.9, group: "attentive", bbox: { x1: 0, y1: 0, x2: 1, y2: 1 } },
        { class_id: 1, class_name: "read", confidence: 0.85, group: "attentive", bbox: { x1: 0, y1: 0, x2: 1, y2: 1 } },
        { class_id: 3, class_name: "discuss", confidence: 0.8, group: "distracted", bbox: { x1: 0, y1: 0, x2: 1, y2: 1 } },
      ],
      summary: { total: 3, attentive: 2, distracted: 1, attentive_percent: 66.7, distracted_percent: 33.3 },
    };

    render(<StatsPanel data={data} />);
    expect(screen.getByText("3")).toBeInTheDocument(); // total
    expect(screen.getByText("2")).toBeInTheDocument(); // attentive
    expect(screen.getByText("1")).toBeInTheDocument(); // distracted
  });
});
