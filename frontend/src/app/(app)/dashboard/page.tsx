"use client";

import { BookOpenText, Flag, ListChecks, PenLine, Search, ShieldCheck } from "lucide-react";

import { useAuth } from "@/components/AuthProvider";
import { NewResearchForm } from "@/components/NewResearchForm";
import { STAGES } from "@/lib/progress";

const ICONS = [ListChecks, Search, BookOpenText, PenLine, ShieldCheck, Flag];

/** Start a new research question; explains what the agents will do. */
export default function DashboardPage() {
  const { user } = useAuth();
  const firstName = user?.full_name?.split(" ")[0];

  return (
    <div className="mx-auto max-w-3xl space-y-10 px-6 py-12">
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 dark:text-white">
          {firstName ? `What should we research, ${firstName}?` : "What should we research?"}
        </h1>
        <p className="text-slate-500 dark:text-slate-400">
          Ask a research question. The agents find real papers, read them, write a cited review and fact-check every claim.
        </p>
      </div>

      <NewResearchForm />

      <section aria-labelledby="how-it-works" className="space-y-4">
        <h2 id="how-it-works" className="text-center text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          How it works
        </h2>
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {STAGES.map((stage, index) => {
            const Icon = ICONS[index];
            return (
              <li key={stage.id} className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
                <div className="mb-2 flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-950/60 dark:text-brand-300">
                    <Icon className="h-4 w-4" aria-hidden />
                  </span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {index + 1}. {stage.label}
                  </span>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{stage.description}</p>
              </li>
            );
          })}
        </ol>
      </section>
    </div>
  );
}
