"use client";

import clsx from "clsx";
import { CheckCircle2, CircleAlert, Loader2, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";

import { useHistory } from "@/components/HistoryProvider";
import { Spinner } from "@/components/ui";
import { formatPercent, formatRelativeTime, sessionLabel } from "@/lib/format";
import type { SessionStatus } from "@/lib/types";

function StatusIcon({ status }: { status: SessionStatus }) {
  if (status === "completed") return <CheckCircle2 className="h-4 w-4 text-emerald-500" aria-label="Completed" />;
  if (status === "failed") return <CircleAlert className="h-4 w-4 text-red-500" aria-label="Failed" />;
  return <Loader2 className="h-4 w-4 animate-spin text-brand-500" aria-label="In progress" />;
}

/** The list of past research sessions, with the "New research" button. */
export function HistorySidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { sessions, loading, error, remove } = useHistory();
  const params = useParams<{ id?: string }>();
  const router = useRouter();
  const [deleting, setDeleting] = useState<string | null>(null);

  async function handleDelete(id: string, label: string) {
    if (!window.confirm(`Delete "${label.slice(0, 80)}"? This cannot be undone.`)) return;
    setDeleting(id);
    try {
      await remove(id);
      if (params.id === id) router.push("/dashboard");
    } catch (caught) {
      window.alert(caught instanceof Error ? caught.message : "Could not delete this session");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="p-3">
        <Link
          href="/dashboard"
          onClick={onNavigate}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-3 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          <Plus className="h-4 w-4" aria-hidden /> New research
        </Link>
      </div>
      <p className="px-4 pb-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">History</p>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-4" aria-label="Research history">
        {loading && (
          <div className="p-3">
            <Spinner label="Loading..." />
          </div>
        )}
        {error && <p className="p-3 text-sm text-red-600 dark:text-red-400">{error}</p>}
        {!loading && !error && sessions.length === 0 && (
          <p className="p-3 text-sm text-slate-500 dark:text-slate-400">Your research sessions will appear here.</p>
        )}
        {sessions.map((session) => {
          const label = sessionLabel(session);
          const active = params.id === session.id;
          return (
            <div
              key={session.id}
              className={clsx(
                "group flex items-start gap-2 rounded-lg px-2 py-2 text-sm",
                active ? "bg-brand-50 dark:bg-brand-900/40" : "hover:bg-slate-100 dark:hover:bg-slate-800/70",
              )}
            >
              <span className="mt-0.5 shrink-0">
                <StatusIcon status={session.status} />
              </span>
              <Link href={`/research/${session.id}`} onClick={onNavigate} className="min-w-0 flex-1">
                <span
                  className={clsx(
                    "line-clamp-2 break-words",
                    active ? "font-medium text-brand-800 dark:text-brand-200" : "text-slate-700 dark:text-slate-200",
                  )}
                >
                  {label}
                </span>
                <span className="mt-0.5 block text-xs text-slate-500 dark:text-slate-400">
                  {formatRelativeTime(session.created_at)}
                  {session.status === "completed" && session.metrics?.faithfulness != null &&
                    ` · ${formatPercent(session.metrics.faithfulness)} supported`}
                  {(session.status === "pending" || session.status === "running") &&
                    ` · ${session.current_stage ?? "queued"}`}
                </span>
              </Link>
              <button
                type="button"
                onClick={() => void handleDelete(session.id, label)}
                disabled={deleting === session.id || session.status === "running" || session.status === "pending"}
                className="shrink-0 rounded p-1 text-slate-400 opacity-0 hover:bg-slate-200 hover:text-red-600 focus:opacity-100 disabled:hidden group-hover:opacity-100 dark:hover:bg-slate-700"
                aria-label={`Delete ${label}`}
                title="Delete"
              >
                <Trash2 className="h-4 w-4" aria-hidden />
              </button>
            </div>
          );
        })}
      </nav>
    </div>
  );
}
