/**
 * Splits report text into plain text and citation markers such as "[1]" or
 * "[2, 5]", plus the "[unverified]" marker the backend adds to flagged claims.
 */

export type TextSegment =
  | { kind: "text"; text: string }
  | { kind: "citation"; numbers: number[] }
  | { kind: "unverified" };

const MARKER_RE = /\[(\d+(?:\s*,\s*\d+)*)\]|\[unverified\]/g;

export function splitCitations(text: string): TextSegment[] {
  const segments: TextSegment[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER_RE)) {
    const index = match.index ?? 0;
    if (index > last) {
      segments.push({ kind: "text", text: text.slice(last, index) });
    }
    if (match[1] !== undefined) {
      segments.push({ kind: "citation", numbers: match[1].split(",").map((part) => Number(part.trim())) });
    } else {
      segments.push({ kind: "unverified" });
    }
    last = index + match[0].length;
  }
  if (last < text.length) {
    segments.push({ kind: "text", text: text.slice(last) });
  }
  return segments;
}
