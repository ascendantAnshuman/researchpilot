import { describe, expect, it } from "vitest";

import { deriveStages, selectedPapers } from "@/lib/progress";
import type { ProgressEvent, ProgressStatus } from "@/lib/types";

function event(stage: string, status: ProgressStatus, message = "", data: Record<string, unknown> = {}): ProgressEvent {
  return { stage, status, message, data, timestamp: "2026-09-28T10:00:00Z" };
}

function states(events: ProgressEvent[], finished = false): Record<string, string> {
  return Object.fromEntries(deriveStages(events, finished).map((stage) => [stage.id, stage.state]));
}

describe("deriveStages", () => {
  it("marks finished, running and waiting stages", () => {
    const events = [
      event("planner", "started"),
      event("planner", "completed", "done", { seconds: 1.5 }),
      event("search", "started"),
      event("search", "progress", "Found 29 unique papers"),
    ];

    expect(states(events)).toMatchObject({ planner: "done", search: "active", reader: "waiting" });
    const search = deriveStages(events, false).find((stage) => stage.id === "search");
    expect(search?.message).toBe("Found 29 unique papers");
    expect(deriveStages(events, false)[0].seconds).toBe(1.5);
  });

  it("counts a second synthesizer run as a revision", () => {
    const events = [
      event("synthesizer", "started"),
      event("synthesizer", "completed"),
      event("critic", "started"),
      event("critic", "completed"),
      event("synthesizer", "started"),
    ];

    const synthesizer = deriveStages(events, false).find((stage) => stage.id === "synthesizer");
    expect(synthesizer?.runs).toBe(2);
    expect(synthesizer?.state).toBe("active");
  });

  it("marks stages that never ran as skipped once the run is finished", () => {
    const events = [event("planner", "completed"), event("search", "completed"), event("finalize", "completed")];

    expect(states(events, true)).toMatchObject({ reader: "skipped", critic: "skipped", finalize: "done" });
  });

  it("collects warnings per stage", () => {
    const events = [event("search", "warning", "Semantic Scholar search failed (HTTP 429)")];

    expect(deriveStages(events, false).find((stage) => stage.id === "search")?.warnings).toEqual([
      "Semantic Scholar search failed (HTTP 429)",
    ]);
  });
});

describe("selectedPapers", () => {
  it("returns the papers from the latest search event", () => {
    const papers = [{ paper_id: "arxiv:1", title: "Paper", score: 0.7 }];
    const events = [event("search", "progress", "Selected the top 1 papers", { papers }), event("reader", "started")];

    expect(selectedPapers(events)).toEqual(papers);
    expect(selectedPapers([])).toEqual([]);
  });
});
