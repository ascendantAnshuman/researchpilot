import { afterEach, describe, expect, it, vi } from "vitest";

import { API_URL, ApiError, apiRequest, createApi, errorMessage } from "@/lib/api";

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("errorMessage", () => {
  it("uses a plain string detail", () => {
    expect(errorMessage({ detail: "Incorrect email or password" }, "fallback")).toBe("Incorrect email or password");
  });

  it("joins FastAPI validation errors and drops the 'Value error' prefix", () => {
    const body = { detail: [{ msg: "Value error, Please ask a complete research question" }, { msg: "Field required" }] };
    expect(errorMessage(body, "fallback")).toBe("Please ask a complete research question; Field required");
  });

  it("falls back when the body has no detail", () => {
    expect(errorMessage("<html>", "Request failed")).toBe("Request failed");
  });
});

describe("apiRequest", () => {
  it("sends JSON with the bearer token and parses the response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { ok: true }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await apiRequest<{ ok: boolean }>("/api/research", { method: "POST", body: { q: 1 }, token: "abc" });

    expect(result).toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/api/research`);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer abc");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.body).toBe('{"q":1}');
  });

  it("returns undefined for 204 No Content", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    await expect(apiRequest("/api/reports/x", { method: "DELETE" })).resolves.toBeUndefined();
  });

  it("raises ApiError with the server's message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(409, { detail: "This research run is running" })));

    await expect(apiRequest("/api/reports/x/export")).rejects.toMatchObject({
      status: 409,
      message: "This research run is running",
    });
  });

  it("explains network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = await apiRequest("/api/health").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(0);
    expect((error as ApiError).message).toContain("Cannot reach");
  });
});

describe("createApi", () => {
  it("calls onUnauthorized when the token is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(401, { detail: "Token has expired" })));
    const onUnauthorized = vi.fn();

    await expect(createApi("expired", onUnauthorized).listReports()).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it("downloads exports with the server's file name", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("# Report", { status: 200, headers: { "Content-Disposition": 'attachment; filename="my-report.md"' } }),
      ),
    );

    const file = await createApi("t").downloadExport("abc", "markdown");

    expect(file.filename).toBe("my-report.md");
    expect(await file.blob.text()).toBe("# Report");
  });

  it("streams research events as typed messages", async () => {
    const body = 'event: running\r\ndata: {"message":"Research started"}\r\n\r\nevent: done\r\ndata: {"status":"completed"}\r\n\r\n';
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const received: string[] = [];

    await createApi("t").streamEvents("abc", (message) => received.push(message.event));

    expect(received).toEqual(["running", "done"]);
  });
});
