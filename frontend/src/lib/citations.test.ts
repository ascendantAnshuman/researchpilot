import { describe, expect, it } from "vitest";

import { splitCitations } from "@/lib/citations";

describe("splitCitations", () => {
  it("separates text, single and grouped citations", () => {
    expect(splitCitations("Grounding helps [1]. Critics help more [2, 5].")).toEqual([
      { kind: "text", text: "Grounding helps " },
      { kind: "citation", numbers: [1] },
      { kind: "text", text: ". Critics help more " },
      { kind: "citation", numbers: [2, 5] },
      { kind: "text", text: "." },
    ]);
  });

  it("recognises the [unverified] marker", () => {
    expect(splitCitations("A bold claim [3]. [unverified]")).toEqual([
      { kind: "text", text: "A bold claim " },
      { kind: "citation", numbers: [3] },
      { kind: "text", text: ". " },
      { kind: "unverified" },
    ]);
  });

  it("leaves other brackets alone", () => {
    expect(splitCitations("Range [a, b] and [E3] stay text.")).toEqual([
      { kind: "text", text: "Range [a, b] and [E3] stay text." },
    ]);
  });

  it("handles empty text", () => {
    expect(splitCitations("")).toEqual([]);
  });
});
