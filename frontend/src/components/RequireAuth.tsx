"use client";

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { useAuth } from "@/components/AuthProvider";
import { Spinner } from "@/components/ui";

/** Renders its children only for logged-in users; everyone else is sent to /login. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, token, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !token) {
      router.replace("/login");
    }
  }, [loading, token, router]);

  if (loading || !token) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner label="Loading ResearchPilot..." />
      </div>
    );
  }
  if (!user) {
    return (
      <div className="flex h-screen items-center justify-center px-6 text-center text-sm text-slate-600 dark:text-slate-300">
        Cannot reach the ResearchPilot server. Make sure the backend is running, then reload this page.
      </div>
    );
  }
  return <>{children}</>;
}
