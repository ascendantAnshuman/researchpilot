import { describe, expect, it } from "vitest";

import { readEventStream, SSEParser, type SSEMessage } from "@/lib/sse";

function parse(chunks: string[]): SSEMessage[] {
  const messages: SSEMessage[] = [];
  const parser = new SSEParser((message) => messages.push(message));
  chunks.forEach((chunk) => parser.push(chunk));
  parser.flush();
  return messages;
}

// Exactly what sse-starlette sends: CRLF line endings, an id, and keep-alive comments.
const STREAM =
  'event: progress\r\ndata: {"stage":"planner"}\r\nid: 0\r\n\r\n' +
  ": ping - 2026-09-28 10:00:00\r\n\r\n" +
  'event: done\r\ndata: {"status":"completed"}\r\nid: 1\r\n\r\n';

describe("SSEParser", () => {
  it("parses events with CRLF line endings and ignores ping comments", () => {
    expect(parse([STREAM])).toEqual([
      { event: "progress", data: '{"stage":"planner"}', id: "0" },
      { event: "done", data: '{"status":"completed"}', id: "1" },
    ]);
  });

  it("gives the same result however the stream is split into chunks", () => {
    const expected = parse([STREAM]);
    for (let cut = 1; cut < STREAM.length; cut += 1) {
      expect(parse([STREAM.slice(0, cut), STREAM.slice(cut)])).toEqual(expected);
    }
  });

  it("handles a stream delivered one character at a time", () => {
    expect(parse([...STREAM])).toEqual(parse([STREAM]));
  });

  it("joins multi-line data and defaults the event name to 'message'", () => {
    expect(parse(["data: first\ndata: second\n\n"])).toEqual([{ event: "message", data: "first\nsecond", id: undefined }]);
  });

  it("dispatches a final event even without a trailing blank line", () => {
    expect(parse(["event: done\ndata: {}"])).toEqual([{ event: "done", data: "{}", id: undefined }]);
  });

  it("does not emit events that have no data", () => {
    expect(parse(["event: lonely\n\n: comment\n\n"])).toEqual([]);
  });
});

describe("readEventStream", () => {
  it("reads a streaming fetch Response", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(STREAM.slice(0, 25)));
        controller.enqueue(encoder.encode(STREAM.slice(25)));
        controller.close();
      },
    });
    const messages: SSEMessage[] = [];

    await readEventStream(new Response(body), (message) => messages.push(message));

    expect(messages.map((message) => message.event)).toEqual(["progress", "done"]);
  });
});
