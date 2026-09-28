/**
 * Server-Sent Events over `fetch`.
 *
 * The browser's built-in `EventSource` cannot send an `Authorization` header,
 * so the research stream is read with `fetch` and parsed here. The parser
 * follows the SSE format: lines of `field: value`, a blank line ends an event,
 * lines starting with ":" are comments (the server's keep-alive pings), and
 * lines may end with "\n", "\r\n" or "\r".
 */

export interface SSEMessage {
  event: string;
  data: string;
  id?: string;
}

export class SSEParser {
  private buffer = "";
  private eventName = "message";
  private dataLines: string[] = [];
  private lastId: string | undefined;

  constructor(private readonly onMessage: (message: SSEMessage) => void) {}

  /** Feed the next piece of text from the stream (it may end mid-line). */
  push(chunk: string): void {
    this.buffer += chunk;
    // A trailing "\r" could be the first half of "\r\n": wait for the next chunk before deciding.
    const holdBack = this.buffer.endsWith("\r") ? 1 : 0;
    const complete = this.buffer.slice(0, this.buffer.length - holdBack).replace(/\r\n?/g, "\n");
    const held = this.buffer.slice(this.buffer.length - holdBack);
    const lines = complete.split("\n");
    const partial = lines.pop() ?? "";
    for (const line of lines) {
      this.processLine(line);
    }
    this.buffer = partial + held;
  }

  /** Call when the stream ends: processes any final line and pending event. */
  flush(): void {
    if (this.buffer) {
      const rest = this.buffer.replace(/\r\n?/g, "\n");
      this.buffer = "";
      for (const line of rest.split("\n")) {
        this.processLine(line);
      }
    }
    this.processLine("");
  }

  private processLine(line: string): void {
    if (line === "") {
      if (this.dataLines.length > 0) {
        this.onMessage({ event: this.eventName, data: this.dataLines.join("\n"), id: this.lastId });
      }
      this.eventName = "message";
      this.dataLines = [];
      return;
    }
    if (line.startsWith(":")) {
      return;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      this.eventName = value || "message";
    } else if (field === "data") {
      this.dataLines.push(value);
    } else if (field === "id") {
      this.lastId = value;
    }
  }
}

/** Read an SSE response body to the end, calling `onMessage` for every event. */
export async function readEventStream(response: Response, onMessage: (message: SSEMessage) => void): Promise<void> {
  if (!response.body) {
    throw new Error("This browser does not support streaming responses.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser(onMessage);
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.flush();
  } finally {
    reader.releaseLock();
  }
}
