/** Small, reusable UI primitives styled with Tailwind (light and dark mode). */

import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-700 disabled:bg-brand-400 dark:disabled:bg-brand-800",
  secondary:
    "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800",
  ghost: "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
  danger: "bg-red-600 text-white hover:bg-red-700",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({ variant = "primary", loading = false, icon, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-950",
        "disabled:cursor-not-allowed",
        BUTTON_STYLES[variant],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

type BadgeTone = "neutral" | "brand" | "green" | "amber" | "red" | "blue";

const BADGE_STYLES: Record<BadgeTone, string> = {
  neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  brand: "bg-brand-100 text-brand-700 dark:bg-brand-900/60 dark:text-brand-200",
  green: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300",
  amber: "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300",
  red: "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300",
  blue: "bg-sky-100 text-sky-800 dark:bg-sky-900/50 dark:text-sky-300",
};

export function Badge({ tone = "neutral", children, className }: { tone?: BadgeTone; children: ReactNode; className?: string }) {
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", BADGE_STYLES[tone], className)}>
      {children}
    </span>
  );
}

export function Spinner({ className, label }: { className?: string; label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400" role="status">
      <Loader2 className={clsx("h-5 w-5 animate-spin", className)} aria-hidden />
      {label ?? <span className="sr-only">Loading</span>}
    </span>
  );
}

type AlertTone = "info" | "success" | "warning" | "error";

const ALERT_STYLES: Record<AlertTone, { box: string; icon: ReactNode }> = {
  info: {
    box: "border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-900 dark:bg-sky-950/50 dark:text-sky-200",
    icon: <Info className="h-4 w-4 shrink-0" aria-hidden />,
  },
  success: {
    box: "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-200",
    icon: <CheckCircle2 className="h-4 w-4 shrink-0" aria-hidden />,
  },
  warning: {
    box: "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-200",
    icon: <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden />,
  },
  error: {
    box: "border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200",
    icon: <XCircle className="h-4 w-4 shrink-0" aria-hidden />,
  },
};

export function Alert({ tone = "info", title, children }: { tone?: AlertTone; title?: string; children?: ReactNode }) {
  const style = ALERT_STYLES[tone];
  return (
    <div className={clsx("flex gap-3 rounded-lg border p-3 text-sm", style.box)} role={tone === "error" ? "alert" : "status"}>
      <span className="mt-0.5">{style.icon}</span>
      <div className="min-w-0 space-y-1">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className="break-words">{children}</div>}
      </div>
    </div>
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900", className)}>
      {children}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-12 text-center text-slate-500 dark:text-slate-400">
      <div className="text-slate-400 dark:text-slate-500">{icon}</div>
      <p className="font-medium text-slate-700 dark:text-slate-200">{title}</p>
      {children && <div className="max-w-md text-sm">{children}</div>}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 px-3 py-2 dark:border-slate-800" title={hint}>
      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">{value}</p>
    </div>
  );
}
