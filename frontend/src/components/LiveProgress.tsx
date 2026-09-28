"use client";

import clsx from "clsx";
import {
  BookOpenText,
  Check,
  FileSearch,
  Flag,
  ListChecks,
  Loader2,
  PenLine,
  Search,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useAuth } from "@/components/AuthProvider";
import { Alert, Badge, Card } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import { deriveStages, selectedPapers, type StageId, type StageView } from "@/lib/progress";
import type { ProgressEvent, StreamMessage } from "@/lib/types";

const STAGE_ICONS: Record<StageId, LucideIcon> = {
  planner: ListChecks,
  search: Search,
  reader: BookOpenText,
  synthesizer: PenLine,
  critic: ShieldCheck,
  finalize: Flag,
};

const MAX_RECONNECTS = 5;

type Phase = "connecting" | "queued" | "running" | "done" | "error";

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      resolve();
    });
  });
}

function StageRow({ stage }: { stage: StageView }) {
  const Icon = STAGE_ICONS[stage.id];
  return (
    <li className="flex gap-3">
      <div
        className={clsx(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-full border",
          stage.state === "done" && "border-emerald-500 bg-emerald-50 text-emerald-600 dark:bg-emerald-950/60",
          stage.state === "active" && "border-brand-500 bg-brand-50 text-brand-600 dark:bg-brand-950/60",
          stage.state === "waiting" && "border-slate-200 text-slate-400 dark:border-slate-700",
          stage.state === "skipped" && "border-dashed border-slate-300 text-slate-300 dark:border-slate-700 dark:text-slate-600",
        )}
      >
        {stage.state === "done" ? (
          <Check className="h-4 w-4" aria-hidden />
        ) : stage.state === "active" ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        ) : (
          <Icon className="h-4 w-4" aria-hidden />
        )}
      </div>
      <div className="min-w-0 flex-1 pb-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className={clsx("font-medium", stage.state === "skipped" ? "text-slate-400" : "text-slate-900 dark:text-slate-100")}>
            {stage.label}
          </span>
          {stage.runs > 1 && <Badge tone="amber">revision {stage.runs - 1}</Badge>}
          {stage.state === "skipped" && <Badge>skipped</Badge>}
          {stage.seconds !== null && stage.state === "done" && (
            <span className="text-xs text-slate-400">{formatDuration(stage.seconds)}</span>
          )}
        </div>
        <p className="text-sm text-slate-500 dark:text-slate-400">{stage.message ?? stage.description}</p>
        {stage.warnings.map((warning) => (
          <p key={warning} className="mt-1 text-xs text-amber-700 dark:text-amber-400">
            {warning}
          </p>
        ))}
      </div>
    </li>
  );
}

/** Live view of a running pipeline, fed by the SSE stream. Calls `onFinished` when the run ends. */
export function LiveProgress({
  sessionId,
  question,
  createdAt,
  onFinished,
}: {
  sessionId: string;
  question: string;
  createdAt: string;
  onFinished: () => void;
}) {
  const { api } = useAuth();
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [phase, setPhase] = useState<Phase>("connecting");
  const [failure, setFailure] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const onFinishedRef = useRef(onFinished);
  const startedAt = new Date(createdAt).getTime();

  useEffect(() => {
    onFinishedRef.current = onFinished;
  }, [onFinished]);

  useEffect(() => {
    const controller = new AbortController();
    let finished = false;

    const handle = (message: StreamMessage) => {
      switch (message.event) {
        case "queued":
          setPhase("queued");
          break;
        case "running":
          setPhase("running");
          break;
        case "progress":
          setPhase("running");
          setEvents((current) => [...current, message.data]);
          break;
        case "done":
          finished = true;
          setPhase("done");
          onFinishedRef.current();
          break;
        case "error":
          finished = true;
          setPhase("error");
          setFailure(message.data.message);
          onFinishedRef.current();
          break;
      }
    };

    async function connect() {
      for (let attempt = 0; attempt <= MAX_RECONNECTS && !finished && !controller.signal.aborted; attempt += 1) {
        if (attempt > 0) {
          await sleep(1500 * attempt, controller.signal);
        }
        setEvents([]); // the server replays the whole history on every connection
        try {
          await api.streamEvents(sessionId, handle, controller.signal);
        } catch (error) {
          if (controller.signal.aborted) return;
          if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
            setPhase("error");
            setFailure(error.message);
            return;
          }
        }
      }
      if (!finished && !controller.signal.aborted) {
        setPhase("error");
        setFailure("Lost the connection to the server. Reload the page to check on this run.");
      }
    }

    void connect();
    return () => controller.abort();
  }, [api, sessionId]);

  useEffect(() => {
    if (phase === "done" || phase === "error") return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [phase]);

  const stages = useMemo(() => deriveStages(events, phase === "done"), [events, phase]);
  const papers = useMemo(() => selectedPapers(events), [events]);
  const log = events.filter((event) => event.status !== "started" && event.status !== "completed");

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div className="space-y-1">
        <p className="text-sm font-medium text-brand-600 dark:text-brand-400">
          {phase === "queued" ? "Waiting in queue" : phase === "done" ? "Finished" : "Researching"} ·{" "}
          {formatDuration((now - startedAt) / 1000)}
        </p>
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">{question}</h1>
      </div>

      {phase === "queued" && <Alert tone="info">Another research run is in progress. Yours starts as soon as it finishes.</Alert>}
      {phase === "error" && failure && <Alert tone="error" title="The research run failed">{failure}</Alert>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <h2 className="mb-4 font-semibold text-slate-900 dark:text-white">Agent pipeline</h2>
          <ol>
            {stages.map((stage) => (
              <StageRow key={stage.id} stage={stage} />
            ))}
          </ol>
        </Card>

        <div className="space-y-6">
          {papers.length > 0 && (
            <Card>
              <h2 className="mb-3 flex items-center gap-2 font-semibold text-slate-900 dark:text-white">
                <FileSearch className="h-4 w-4" aria-hidden /> Selected papers
              </h2>
              <ol className="space-y-2 text-sm">
                {papers.map((paper, index) => (
                  <li key={paper.paper_id} className="flex gap-2 text-slate-700 dark:text-slate-300">
                    <span className="w-5 shrink-0 text-right text-slate-400">{index + 1}.</span>
                    <span className="min-w-0 flex-1">{paper.title}</span>
                    <span className="shrink-0 text-xs text-slate-400">{paper.score.toFixed(2)}</span>
                  </li>
                ))}
              </ol>
            </Card>
          )}
          <Card>
            <h2 className="mb-3 font-semibold text-slate-900 dark:text-white">Activity</h2>
            {log.length === 0 ? (
              <p className="text-sm text-slate-500 dark:text-slate-400">Connecting to the agents...</p>
            ) : (
              <ul className="max-h-96 space-y-1.5 overflow-y-auto text-sm" aria-live="polite">
                {log.map((event, index) => (
                  <li key={`${event.timestamp}-${index}`} className="flex gap-2">
                    <span className="w-20 shrink-0 font-mono text-xs leading-5 text-slate-400">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                    <span
                      className={clsx(
                        "min-w-0 flex-1",
                        event.status === "warning" ? "text-amber-700 dark:text-amber-400" : "text-slate-700 dark:text-slate-300",
                      )}
                    >
                      <span className="font-medium capitalize">{event.stage}:</span> {event.message}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
