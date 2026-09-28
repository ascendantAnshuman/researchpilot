import { Compass } from "lucide-react";

/** The ResearchPilot word mark. */
export function Logo() {
  return (
    <span className="inline-flex items-center gap-2 font-semibold text-slate-900 dark:text-white">
      <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600 text-white">
        <Compass className="h-4 w-4" aria-hidden />
      </span>
      ResearchPilot
    </span>
  );
}
