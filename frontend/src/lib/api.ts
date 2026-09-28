/**
 * Typed client for the ResearchPilot FastAPI backend.
 *
 * `createApi(token)` returns functions bound to the logged-in user's JWT. Any
 * 401 response triggers `onUnauthorized` (the AuthProvider logs the user out).
 */

import { readEventStream } from "@/lib/sse";
import type {
  ChatExchange,
  ChatMessage,
  ExportFormat,
  ResearchOptions,
  ResearchStarted,
  SessionDetail,
  SessionSummary,
  StreamMessage,
  TokenResponse,
  User,
} from "@/lib/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const NETWORK_ERROR = "Cannot reach the ResearchPilot server. Is the backend running?";

/** Turn a FastAPI error body (string or validation list) into one readable sentence. */
export function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) => {
          const message = item && typeof item === "object" && "msg" in item ? String(item.msg) : String(item);
          return message.replace(/^Value error, /, "");
        })
        .join("; ");
    }
  }
  return fallback;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
  signal?: AbortSignal;
}

/** Send a JSON request and return the parsed JSON response (undefined for 204). */
export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }
  let body: string | undefined;
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: options.method ?? (body !== undefined ? "POST" : "GET"),
      headers,
      body,
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (isAbort(error)) {
      throw error;
    }
    throw new ApiError(0, NETWORK_ERROR);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, errorMessage(parsed, `Request failed (HTTP ${response.status}).`));
  }
  return parsed as T;
}

export function login(email: string, password: string): Promise<TokenResponse> {
  return apiRequest<TokenResponse>("/api/auth/login", { method: "POST", body: { email, password } });
}

export function register(email: string, password: string, fullName?: string): Promise<TokenResponse> {
  return apiRequest<TokenResponse>("/api/auth/register", {
    method: "POST",
    body: { email, password, full_name: fullName?.trim() || null },
  });
}

export interface DownloadedFile {
  blob: Blob;
  filename: string;
}

/** Functions bound to a user's token; 401 responses call `onUnauthorized`. */
export function createApi(token: string | null, onUnauthorized: () => void = () => undefined) {
  async function call<T>(path: string, options: Omit<RequestOptions, "token"> = {}): Promise<T> {
    try {
      return await apiRequest<T>(path, { ...options, token });
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        onUnauthorized();
      }
      throw error;
    }
  }

  async function authorisedFetch(path: string, accept: string, signal?: AbortSignal): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(`${API_URL}${path}`, {
        headers: { Accept: accept, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        signal,
        cache: "no-store",
      });
    } catch (error) {
      if (isAbort(error)) {
        throw error;
      }
      throw new ApiError(0, NETWORK_ERROR);
    }
    if (!response.ok) {
      if (response.status === 401) {
        onUnauthorized();
      }
      let parsed: unknown = null;
      try {
        parsed = await response.json();
      } catch {
        parsed = null;
      }
      throw new ApiError(response.status, errorMessage(parsed, `Request failed (HTTP ${response.status}).`));
    }
    return response;
  }

  return {
    me: () => call<User>("/api/auth/me"),
    startResearch: (options: ResearchOptions) =>
      call<ResearchStarted>("/api/research", { method: "POST", body: options }),
    listReports: () => call<SessionSummary[]>("/api/reports"),
    getReport: (id: string) => call<SessionDetail>(`/api/reports/${encodeURIComponent(id)}`),
    deleteReport: (id: string) => call<void>(`/api/reports/${encodeURIComponent(id)}`, { method: "DELETE" }),
    chatHistory: (id: string) => call<ChatMessage[]>(`/api/chat/${encodeURIComponent(id)}`),
    ask: (id: string, message: string) =>
      call<ChatExchange>(`/api/chat/${encodeURIComponent(id)}`, { method: "POST", body: { message } }),

    /** Download the report; returns the file and the server's suggested file name. */
    async downloadExport(id: string, format: ExportFormat): Promise<DownloadedFile> {
      const response = await authorisedFetch(
        `/api/reports/${encodeURIComponent(id)}/export?format=${format}`,
        format === "pdf" ? "application/pdf" : "text/markdown",
      );
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const match = /filename="([^"]+)"/.exec(disposition);
      return { blob: await response.blob(), filename: match?.[1] ?? `report.${format === "pdf" ? "pdf" : "md"}` };
    },

    /** Stream a run's live progress until the server closes the stream. */
    async streamEvents(id: string, onMessage: (message: StreamMessage) => void, signal?: AbortSignal): Promise<void> {
      const response = await authorisedFetch(
        `/api/research/${encodeURIComponent(id)}/events`,
        "text/event-stream",
        signal,
      );
      await readEventStream(response, (message) => {
        try {
          onMessage({ event: message.event, data: JSON.parse(message.data) } as StreamMessage);
        } catch {
          // Ignore malformed events rather than breaking the whole stream.
        }
      });
    },
  };
}

export type Api = ReturnType<typeof createApi>;
