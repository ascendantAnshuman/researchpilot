"use client";

import { useParams } from "next/navigation";

import { SessionView } from "@/components/SessionView";

/** One research session: live progress, the finished report, or the failure reason. */
export default function ResearchPage() {
  const params = useParams<{ id: string }>();
  return <SessionView key={params.id} sessionId={params.id} />;
}
