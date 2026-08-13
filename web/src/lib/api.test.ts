import { describe, expect, it, vi } from "vitest";
import { api } from "./api";

describe("API client", () => {
  it("identifies browser writes to the local service", async () => {
    const fetchMock = vi.fn<typeof fetch>(() =>
      Promise.resolve(new Response(JSON.stringify({ id: "project_1" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.createProject("https://www.youtube.com/watch?v=abc123");

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.headers).toMatchObject({ "X-ClipForge-Client": "web" });
  });

  it("routes opt-in live provider tests through the protected write client", async () => {
    const fetchMock = vi.fn<typeof fetch>(() => Promise.resolve(new Response(JSON.stringify({ status: "healthy" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));
    vi.stubGlobal("fetch", fetchMock);

    await api.testProvider("profile_openrouter", true);

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/provider-profiles/profile_openrouter/test?live=true");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.headers).toMatchObject({ "X-ClipForge-Client": "web" });
  });

  it("deletes a terminal job through the protected write client", async () => {
    const fetchMock = vi.fn<typeof fetch>(() => Promise.resolve(new Response(null, { status: 204 })));
    vi.stubGlobal("fetch", fetchMock);

    await api.deleteJob("job_complete");

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/jobs/job_complete");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("DELETE");
    expect(init.headers).toMatchObject({ "X-ClipForge-Client": "web" });
  });

  it("requests bounded recent-project pages with an opaque cursor", async () => {
    const fetchMock = vi.fn<typeof fetch>(() => Promise.resolve(new Response(JSON.stringify({ items: [], nextCursor: null }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));
    vi.stubGlobal("fetch", fetchMock);

    await api.projects("opaque cursor", 24);

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/projects?limit=24&cursor=opaque+cursor");
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
  });

  it("discards an abandoned project through the protected write client", async () => {
    const fetchMock = vi.fn<typeof fetch>(() => Promise.resolve(new Response(null, { status: 204 })));
    vi.stubGlobal("fetch", fetchMock);

    await api.deleteProject("project_empty");

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("/projects/project_empty");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("DELETE");
    expect(init.headers).toMatchObject({ "X-ClipForge-Client": "web" });
  });
});
