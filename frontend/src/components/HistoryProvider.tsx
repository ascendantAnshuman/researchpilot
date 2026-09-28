"use client";

/**
 * The user's research history (sidebar). Refreshed on demand and polled every
 * few seconds while any run is still pending or running.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { useAuth } from "@/components/AuthProvider";
import type { SessionSummary } from "@/lib/types";

const POLL_MS = 4000;

interface HistoryContextValue {
  sessions: SessionSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const HistoryContext = createContext<HistoryContextValue | null>(null);

export function HistoryProvider({ children }: { children: ReactNode }) {
  const { api, user } = useAuth();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!user) return;
    try {
      setSessions(await api.listReports());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load your history");
    } finally {
      setLoading(false);
    }
  }, [api, user]);

  const remove = useCallback(
    async (id: string) => {
      await api.deleteReport(id);
      setSessions((current) => current.filter((session) => session.id !== id));
    },
    [api],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const hasActiveRun = sessions.some((session) => session.status === "pending" || session.status === "running");
  useEffect(() => {
    if (!hasActiveRun) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [hasActiveRun, refresh]);

  const value = useMemo(() => ({ sessions, loading, error, refresh, remove }), [sessions, loading, error, refresh, remove]);
  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>;
}

export function useHistory(): HistoryContextValue {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error("useHistory must be used inside HistoryProvider");
  }
  return context;
}
