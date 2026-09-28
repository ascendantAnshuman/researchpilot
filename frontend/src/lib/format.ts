/** Small formatting helpers for dates, durations, money and percentages. */

export function formatRelativeTime(iso: string, now: Date = new Date()): string {
  const seconds = Math.round((now.getTime() - new Date(iso).getTime()) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days} d ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return minutes > 0 ? `${minutes}m ${rest.toString().padStart(2, "0")}s` : `${rest}s`;
}

export function formatCost(usd: number): string {
  return usd < 0.01 && usd > 0 ? "<$0.01" : `$${usd.toFixed(2)}`;
}

export function formatPercent(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : `${Math.round(value * 100)}%`;
}

export function formatAuthors(authors: string[], max = 3): string {
  if (authors.length === 0) return "Unknown authors";
  return authors.length > max ? `${authors.slice(0, max).join(", ")} et al.` : authors.join(", ");
}

export function sessionLabel(session: { title: string | null; question: string }): string {
  return session.title || session.question;
}
