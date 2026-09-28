import { describe, expect, it } from "vitest";

import { formatAuthors, formatCost, formatDuration, formatPercent, formatRelativeTime } from "@/lib/format";

describe("format helpers", () => {
  it("formats durations", () => {
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(42.4)).toBe("42s");
    expect(formatDuration(185)).toBe("3m 05s");
  });

  it("formats cost and percentages", () => {
    expect(formatCost(1.234)).toBe("$1.23");
    expect(formatCost(0.004)).toBe("<$0.01");
    expect(formatCost(0)).toBe("$0.00");
    expect(formatPercent(0.857)).toBe("86%");
    expect(formatPercent(null)).toBe("n/a");
  });

  it("formats relative times", () => {
    const now = new Date("2026-09-28T12:00:00Z");
    expect(formatRelativeTime("2026-09-28T11:59:40Z", now)).toBe("just now");
    expect(formatRelativeTime("2026-09-28T11:30:00Z", now)).toBe("30 min ago");
    expect(formatRelativeTime("2026-09-28T07:00:00Z", now)).toBe("5 h ago");
    expect(formatRelativeTime("2026-09-26T12:00:00Z", now)).toBe("2 d ago");
  });

  it("shortens long author lists", () => {
    expect(formatAuthors([])).toBe("Unknown authors");
    expect(formatAuthors(["A", "B"])).toBe("A, B");
    expect(formatAuthors(["A", "B", "C", "D"])).toBe("A, B, C et al.");
  });
});
