import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex h-screen flex-col items-center justify-center gap-3 px-6 text-center">
      <p className="text-5xl font-semibold text-slate-300 dark:text-slate-700">404</p>
      <p className="text-slate-600 dark:text-slate-300">This page does not exist.</p>
      <Link href="/dashboard" className="font-medium text-brand-600 hover:underline dark:text-brand-400">
        Back to ResearchPilot
      </Link>
    </div>
  );
}
