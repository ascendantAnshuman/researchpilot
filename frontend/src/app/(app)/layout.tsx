"use client";

import type { ReactNode } from "react";

import { AppShell } from "@/components/AppShell";
import { HistoryProvider } from "@/components/HistoryProvider";
import { RequireAuth } from "@/components/RequireAuth";

/** Every page in this group requires login and shows the history sidebar. */
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <HistoryProvider>
        <AppShell>{children}</AppShell>
      </HistoryProvider>
    </RequireAuth>
  );
}
