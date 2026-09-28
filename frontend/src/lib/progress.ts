/**
 * Turns the stream of progress events into the state of each pipeline stage
 * for the live progress view.
 */

import type { ProgressEvent } from "@/lib/types";

export const STAGES = [
  { id: "planner", label: "Planner", description: "Splits the question into focused sub-queries" },
  { id: "search", label: "Search", description: "Finds and ranks papers on arXiv and Semantic Scholar" },
  { id: "reader", label: "Reader", description: "Downloads PDFs, chunks and embeds them, extracts key facts" },
  { id: "synthesizer", label: "Synthesizer", description: "Writes the review using only retrieved passages" },
  { id: "critic", label: "Critic", description: "Checks every claim against the sources" },
  { id: "finalize", label: "Finalize", description: "Numbers citations and assembles the report" },
] as const;

export type StageId = (typeof STAGES)[number]["id"];
export type StageState = "waiting" | "active" | "done" | "skipped";

export interface StageView {
  id: StageId;
  label: string;
  description: string;
  state: StageState;
  message: string | null;
  runs: number;
  seconds: number | null;
  warnings: string[];
}

/** Compute every stage's state from the events received so far. */
export function deriveStages(events: ProgressEvent[], finished: boolean): StageView[] {
  return STAGES.map((stage) => {
    const own = events.filter((event) => event.stage === stage.id);
    const runs = own.filter((event) => event.status === "started").length;
    const last = own[own.length - 1];
    const completions = own.filter((event) => event.status === "completed");
    const lastCompletion = completions[completions.length - 1];
    let state: StageState = "waiting";
    if (last) {
      state = last.status === "completed" ? "done" : "active";
    } else if (finished) {
      state = "skipped";
    }
    if (finished && state === "active") {
      state = "done";
    }
    const secondsValue = lastCompletion?.data.seconds;
    return {
      id: stage.id,
      label: stage.label,
      description: stage.description,
      state,
      message: own.filter((event) => event.status !== "started" && event.status !== "completed").at(-1)?.message ?? null,
      runs,
      seconds: typeof secondsValue === "number" ? secondsValue : null,
      warnings: own.filter((event) => event.status === "warning").map((event) => event.message),
    };
  });
}

export interface SelectedPaper {
  paper_id: string;
  title: string;
  score: number;
}

/** Papers the Search agent selected, if that event has arrived. */
export function selectedPapers(events: ProgressEvent[]): SelectedPaper[] {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const papers = events[index].data.papers;
    if (events[index].stage === "search" && Array.isArray(papers)) {
      return papers as SelectedPaper[];
    }
  }
  return [];
}
