"use client";

import { RotateCcw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { useAuth } from "@/components/AuthProvider";
import { useHistory } from "@/components/HistoryProvider";
import { LiveProgress } from "@/components/LiveProgress";
import { ReportView } from "@/components/report/ReportView";
import { Alert, Button, Spinner } from "@/components/ui";
import type { SessionDetail } from "@/lib/types";

/** Shows a session: live progress while it runs, the report when done, or the failure reason. */
export function SessionView({ sessionId }: { sessionId: string }) {
  const { api } = useAuth();
  const { refresh } = useHistory();
  const router = useRouter();
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);

  const load = useCallback(async () => {
    try {
      setDetail(await api.getReport(sessionId));
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load this session");
    }
  }, [api, sessionId]);

  useEffect(() => {
    setDetail(null);
    void load();
  }, [load]);

  const handleFinished = useCallback(() => {
    void load();
    void refresh();
  }, [load, refresh]);

  async function retry(question: string) {
    setRetrying(true);
    try {
      const started = await api.startResearch({ question });
      void refresh();
      router.push(`/research/${started.session.id}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not restart the research");
      setRetrying(false);
    }
  }

  if (error && !detail) {
    return (
      <div className="mx-auto max-w-3xl p-6">
        <Alert tone="error" title="Could not open this research session">
          {error}
        </Alert>
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="flex h-full items-center justify-center p-12">
        <Spinner label="Loading..." />
      </div>
    );
  }
  if (detail.status === "completed" && detail.report) {
    return <ReportView sessionId={detail.id} report={detail.report} />;
  }
  if (detail.status === "failed") {
    return (
      <div className="mx-auto max-w-3xl space-y-4 p-6">
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">{detail.question}</h1>
        <Alert tone="error" title="This research run failed">
          {detail.error_message ?? "Unknown error."}
        </Alert>
        {error && <Alert tone="error">{error}</Alert>}
        <Button onClick={() => void retry(detail.question)} loading={retrying} icon={<RotateCcw className="h-4 w-4" aria-hidden />}>
          Run it again
        </Button>
      </div>
    );
  }
  return <LiveProgress sessionId={detail.id} question={detail.question} createdAt={detail.created_at} onFinished={handleFinished} />;
}
