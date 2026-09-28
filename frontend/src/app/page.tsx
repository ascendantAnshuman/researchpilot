"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { useAuth } from "@/components/AuthProvider";
import { Spinner } from "@/components/ui";

/** Entry point: send logged-in users to the dashboard and everyone else to the login page. */
export default function HomePage() {
  const { token, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading) {
      router.replace(token ? "/dashboard" : "/login");
    }
  }, [loading, token, router]);

  return (
    <div className="flex h-screen items-center justify-center">
      <Spinner label="Loading ResearchPilot..." />
    </div>
  );
}
