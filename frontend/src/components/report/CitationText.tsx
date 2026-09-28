import { Fragment } from "react";

import { splitCitations } from "@/lib/citations";

/**
 * Renders text with its citation markers as links.
 * "[1, 3]" becomes superscript links to #ref-1 and #ref-3 (or to `anchorPrefix`),
 * and "[unverified]" becomes an amber badge.
 */
export function CitationText({
  text,
  titles,
  anchorPrefix = "ref-",
}: {
  text: string;
  titles?: Map<number, string>;
  anchorPrefix?: string;
}) {
  return (
    <>
      {splitCitations(text).map((segment, index) => {
        if (segment.kind === "text") {
          return <Fragment key={index}>{segment.text}</Fragment>;
        }
        if (segment.kind === "unverified") {
          return (
            <span
              key={index}
              className="ml-1 rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/50 dark:text-amber-300"
              title="The Critic could not verify this sentence against the sources"
            >
              unverified
            </span>
          );
        }
        return (
          <sup key={index} className="ml-0.5 whitespace-nowrap">
            [
            {segment.numbers.map((number, position) => (
              <Fragment key={number}>
                {position > 0 && ", "}
                <a
                  href={`#${anchorPrefix}${number}`}
                  className="font-medium text-brand-600 hover:underline dark:text-brand-400"
                  title={titles?.get(number)}
                >
                  {number}
                </a>
              </Fragment>
            ))}
            ]
          </sup>
        );
      })}
    </>
  );
}
