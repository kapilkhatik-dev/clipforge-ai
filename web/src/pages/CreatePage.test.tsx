import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreatePage } from "./CreatePage";

const response = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  }));

class FakeEventSource {
  static latest: FakeEventSource | null = null;
  readonly url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.latest = this;
  }
}

function WorkspaceRouteHarness() {
  const navigate = useNavigate();
  return (
    <>
      <button type="button" onClick={() => navigate("/create")}>Start a new project</button>
      <Routes>
        <Route path="/projects/:projectId" element={<CreatePage />} />
        <Route path="/create" element={<CreatePage />} />
      </Routes>
    </>
  );
}

describe("CreatePage", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/bootstrap")) {
        return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      }
      if (url.endsWith("/provider-profiles")) {
        return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      }
      if (url.includes("/projects?")) {
        return response({ items: [], nextCursor: null });
      }
      return response({});
    }));
  });

  it("defaults to all AI-selected clips and exposes a bounded limit", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><CreatePage /></MemoryRouter>);

    expect(screen.getByRole("radio", { name: /all best clips/i })).toBeChecked();
    expect(screen.queryByRole("slider", { name: /maximum number of clips/i })).not.toBeInTheDocument();

    await user.click(screen.getByText("Limit output"));

    expect(screen.getByRole("radio", { name: /limit output/i })).toBeChecked();
    expect(screen.getByRole("slider", { name: /maximum number of clips/i })).toHaveAttribute("max", "20");
  });

  it("shows an explicit empty state for the local project history", async () => {
    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    expect(await screen.findByText("No saved generations yet")).toBeInTheDocument();
  });

  it("lists recent projects and reopens their latest generation", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.includes("/projects?")) return response({
        items: [{
          id: "project_recent",
          createdAt: "2026-08-13T10:00:00Z",
          source: { kind: "youtube", url: "https://www.youtube.com/watch?v=recentvideo" },
          generationCount: 3,
          latestGeneration: {
            id: "job_recent",
            status: "completed",
            createdAt: "2026-08-13T10:01:00Z",
            finishedAt: "2026-08-13T10:02:00Z",
            title: "Three perfect punchlines",
            thumbnailUrl: "/api/v1/assets/thumb_recent",
            exportCount: 4,
          },
        }],
        nextCursor: null,
      });
      return response({});
    }));

    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    const card = await screen.findByRole("link", { name: /three perfect punchlines/i });
    expect(card).toHaveAttribute("href", "/projects/project_recent?job=job_recent");
    expect(screen.getByText("3 generations")).toBeInTheDocument();
    expect(screen.getByText(/4 exports/i)).toBeInTheDocument();
  });

  it("requires a valid public URL before submitting", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    await screen.findByText("Codex");
    await user.type(screen.getByLabelText("Video URL"), "ftp://example.com/video");
    await user.click(screen.getByRole("button", { name: /generate clips/i }));
    expect(await screen.findByText(/complete public YouTube video URL/i)).toBeInTheDocument();
  });

  it("discards an empty project when generation setup fails synchronously", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.includes("/projects?")) return response({ items: [], nextCursor: null });
      if (requestUrl.endsWith("/projects") && init?.method === "POST") return response({ id: "project_abandoned" });
      if (requestUrl.endsWith("/projects/project_abandoned/generations")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "Provider is no longer ready" }), {
          status: 422,
          headers: { "Content-Type": "application/json" },
        }));
      }
      if (requestUrl.endsWith("/projects/project_abandoned") && init?.method === "DELETE") {
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    await user.type(screen.getByLabelText("Video URL"), "https://www.youtube.com/watch?v=abcdefghijk");
    await user.click(await screen.findByRole("button", { name: /generate clips/i }));

    expect(await screen.findByText("Provider is no longer ready")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input).endsWith("/projects/project_abandoned") && init?.method === "DELETE",
    )).toBe(true);
  });

  it("adds a rerun to the current project when its source URL is unchanged", async () => {
    const user = userEvent.setup();
    const existingJob = {
      id: "job_existing",
      projectId: "project_reuse",
      status: "completed",
      createdAt: "2026-08-13T10:00:00Z",
      result: { clips: [] },
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.endsWith("/projects/project_reuse") && !init?.method) return response({
        id: "project_reuse",
        createdAt: "2026-08-13T09:00:00Z",
        source: { kind: "youtube", url: "https://www.youtube.com/watch?v=reusesource" },
        jobs: [existingJob],
      });
      if (requestUrl.endsWith("/projects/project_reuse/generations") && init?.method === "POST") return response({
        id: "job_rerun",
        projectId: "project_reuse",
        status: "completed",
        createdAt: "2026-08-13T11:00:00Z",
        result: { clips: [] },
      });
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter initialEntries={[{
        pathname: "/projects/project_reuse",
        search: "?job=job_existing",
        state: { job: existingJob },
      }]}>
        <Routes><Route path="/projects/:projectId" element={<CreatePage />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByDisplayValue("https://www.youtube.com/watch?v=reusesource")).toBeInTheDocument();
    const generate = await screen.findByRole("button", { name: /generate clips/i });
    await waitFor(() => expect(generate).toBeEnabled());
    await user.click(generate);

    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input).endsWith("/projects/project_reuse/generations") && init?.method === "POST",
    )).toBe(true));
    expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input).endsWith("/projects") && init?.method === "POST",
    )).toBe(false);
  });

  it("matches the backend's exact YouTube host allowlist", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    await screen.findByText("Codex");
    const input = screen.getByLabelText("Video URL");
    await user.type(input, "https://evil.youtube.com/watch?v=abc123");
    await user.click(screen.getByRole("button", { name: /generate clips/i }));
    expect(await screen.findByText(/complete public YouTube video URL/i)).toBeInTheDocument();
  });

  it("blocks generation until the active provider has a credential", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/bootstrap")) {
        return response({ defaultProviderProfileId: "profile_openrouter", capabilities: {} });
      }
      if (url.endsWith("/provider-profiles")) {
        return response([{ id: "profile_openrouter", providerId: "openrouter", name: "OpenRouter", model: "openrouter/auto", active: true, generationReady: false, credential: { configured: false } }]);
      }
      return response({});
    }));

    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    expect(await screen.findByRole("button", { name: /generate clips/i })).toBeDisabled();
    expect(await screen.findByText(/configure your AI provider in/i)).toBeInTheDocument();
    expect(screen.getByText(".env")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /refresh provider/i })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /provider setup/i })).not.toBeInTheDocument();
  });

  it("explains the montage format and exposes every supported limit", () => {
    render(<MemoryRouter><CreatePage /></MemoryRouter>);

    expect(screen.getByText("1080 × 1920")).toBeInTheDocument();
    expect(screen.getByText("3–6 sec")).toBeInTheDocument();
    expect(screen.getByText("2–20")).toBeInTheDocument();
    expect(screen.getByText("60 sec")).toBeInTheDocument();
    expect(screen.getByLabelText("Highlight moment size")).toHaveAttribute("min", "3");
    expect(screen.getByLabelText("Highlight moment size")).toHaveAttribute("max", "6");
    expect(screen.getByLabelText("Maximum montage duration")).toHaveAttribute("max", "60");
    expect(screen.getByLabelText("Maximum montage moments")).toHaveAttribute("min", "2");
    expect(screen.getByLabelText("Maximum montage moments")).toHaveAttribute("max", "20");
  });

  it("presents the generated montage as the featured export", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={[{
        pathname: "/create",
        state: {
          job: {
            id: "job_montage",
            projectId: "project_montage",
            status: "completed",
            result: {
              montage: {
                id: "montage_1",
                title: "The complete highlight reel",
                start: 0,
                end: 48,
                duration: 48,
                assets: { videoUrl: "/api/v1/assets/montage_1" },
                editDecisionList: {
                  version: 1,
                  kind: "montage",
                  sourceVideoId: "source_1",
                  sourceRanges: [{ start: 5, end: 9 }, { start: 21, end: 25 }, { start: 40, end: 44 }],
                  videoLayout: "fill-crop",
                  captionPreset: "social-bold",
                },
              },
              clips: [],
            },
          },
        },
      }]}>
        <CreatePage />
      </MemoryRouter>,
    );

    const spotlight = await screen.findByRole("region", { name: "Best moments montage" });
    expect(spotlight).toHaveTextContent("Featured export");
    expect(spotlight).toHaveTextContent("1080 × 1920 · 9:16 MP4");
    expect(spotlight).toHaveTextContent("Up to 60 seconds · 20 moments");
    expect(spotlight).toHaveTextContent("0:48 total");
    expect(spotlight).toHaveTextContent("3 moments");
    const previewButton = screen.getByRole("button", { name: "Preview The complete highlight reel" });
    await user.click(previewButton);
    const dialog = screen.getByRole("dialog", { name: "The complete highlight reel" });
    expect(within(dialog).getByText("Moments")).toBeInTheDocument();
    expect(within(dialog).getByText("3 moments")).toBeInTheDocument();
    expect(within(dialog).queryByText("Source range")).not.toBeInTheDocument();
  });

  it("uses safely stored processing defaults", () => {
    localStorage.setItem("clipforge.processingDefaults", JSON.stringify({
      contentType: "comedy",
      clipCountMode: "limit",
      clipCount: 9,
      minDuration: 15,
      maxDuration: 45,
      layout: "fit-blur",
      montage: false,
    }));
    render(<MemoryRouter><CreatePage /></MemoryRouter>);
    expect(screen.getByRole("radio", { name: /limit output/i })).toBeChecked();
    expect(screen.getByRole("slider", { name: /maximum number of clips/i })).toHaveValue("9");
    expect(screen.getByLabelText("Creative direction")).toHaveValue("comedy");
    expect(screen.getByLabelText("Maximum clip duration")).toHaveValue(45);
  });

  it("closes the preview with Escape and restores focus to its trigger", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={[{
        pathname: "/create",
        state: {
          job: {
            id: "job_ready",
            projectId: "project_ready",
            status: "completed",
            result: {
              clips: [{
                id: "clip_1",
                title: "A sharp punchline",
                start: 10,
                end: 25,
                duration: 15,
                assets: { videoUrl: "/api/v1/assets/video_1" },
              }],
            },
          },
        },
      }]}>
        <CreatePage />
      </MemoryRouter>,
    );

    const trigger = await screen.findByRole("button", { name: "Preview A sharp punchline" });
    await user.click(trigger);
    expect(screen.getByRole("dialog", { name: "A sharp punchline" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close preview" })).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("applies SSE progress locally and locks the current recipe", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    render(
      <MemoryRouter initialEntries={[{
        pathname: "/create",
        state: {
          job: {
            id: "job_running",
            projectId: "project_running",
            status: "running",
            stage: "download",
            stageProgress: 0.1,
            result: null,
          },
        },
      }]}>
        <CreatePage />
      </MemoryRouter>,
    );

    expect(await screen.findByRole("button", { name: /generation in progress/i })).toBeDisabled();
    expect(screen.getByRole("radio", { name: /all best clips/i })).toBeDisabled();
    expect(FakeEventSource.latest).not.toBeNull();

    FakeEventSource.latest?.onmessage?.(new MessageEvent("message", {
      data: JSON.stringify({
        jobId: "job_running",
        sequence: 3,
        type: "progress",
        event: { stage: "analyze", message: "Finding the funniest moments", progress: 0.4 },
      }),
    }));

    expect(await screen.findByText("Finding the funniest moments")).toBeInTheDocument();
    expect(screen.getByText("40%", { selector: ".progress-value" })).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([input]) => String(input).includes("/jobs/"))).toBe(false);
  });

  it("rejects a restored job from a different project route", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) {
        return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      }
      if (requestUrl.endsWith("/provider-profiles")) {
        return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      }
      if (requestUrl.endsWith("/jobs/job_wrong")) {
        return response({ id: "job_wrong", projectId: "project_other", status: "completed", result: { clips: [] } });
      }
      return response({});
    }));

    render(
      <MemoryRouter initialEntries={["/projects/project_expected?job=job_wrong"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<CreatePage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText(/does not belong to the requested project/i)).toBeInTheDocument();
  });

  it("restores the newest generation from a project route without a job query", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) {
        return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      }
      if (requestUrl.endsWith("/provider-profiles")) {
        return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      }
      if (requestUrl.endsWith("/projects/project_saved")) {
        return response({
          id: "project_saved",
          createdAt: "2026-08-13T00:00:00Z",
          source: { kind: "youtube", url: "https://www.youtube.com/watch?v=saved" },
          jobs: [{ id: "job_saved", projectId: "project_saved", status: "completed", result: { clips: [] } }],
        });
      }
      return response({});
    }));

    render(
      <MemoryRouter initialEntries={["/projects/project_saved"]}>
        <Routes>
          <Route path="/projects/:projectId" element={<CreatePage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByDisplayValue("https://www.youtube.com/watch?v=saved")).toBeInTheDocument();
    expect(screen.getByText(/generation completed without exported clips/i)).toBeInTheDocument();
  });

  it("clears project results when navigating from a project back to create", async () => {
    const user = userEvent.setup();
    const previousJob = {
      id: "job_previous",
      projectId: "project_previous",
      status: "completed",
      result: { clips: [{ id: "clip_previous", title: "Previous project result", start: 0, end: 12, duration: 12, assets: { videoUrl: "/previous.mp4" } }] },
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.endsWith("/projects/project_previous")) return response({
        id: "project_previous",
        createdAt: "2026-08-13T09:00:00Z",
        source: { kind: "youtube", url: "https://www.youtube.com/watch?v=previous123" },
        jobs: [previousJob],
      });
      if (requestUrl.includes("/projects?")) return response({ items: [], nextCursor: null });
      return response({});
    }));

    render(
      <MemoryRouter initialEntries={[{
        pathname: "/projects/project_previous",
        search: "?job=job_previous",
        state: { job: previousJob },
      }]}>
        <WorkspaceRouteHarness />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Previous project result")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("https://www.youtube.com/watch?v=previous123")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start a new project" }));

    expect(await screen.findByText("Your best moments will land here")).toBeInTheDocument();
    expect(screen.getByLabelText("Video URL")).toHaveValue("");
    expect(screen.queryByText("Previous project result")).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Project generation history" })).not.toBeInTheDocument();
  });

  it("clears stale results while a requested job restore fails", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.endsWith("/jobs/job_missing")) {
        return Promise.resolve(new Response(JSON.stringify({ detail: "Job not found" }), { status: 404, headers: { "Content-Type": "application/json" } }));
      }
      return response({});
    }));

    render(
      <MemoryRouter initialEntries={[{
        pathname: "/projects/project_1",
        search: "?job=job_missing",
        state: {
          job: {
            id: "job_old",
            projectId: "project_1",
            status: "completed",
            result: { clips: [{ id: "old_clip", title: "Old result", start: 0, end: 10, duration: 10, assets: { videoUrl: "/old.mp4" } }] },
          },
        },
      }]}>
        <Routes><Route path="/projects/:projectId" element={<CreatePage />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByText("Job not found")).toBeInTheDocument();
    expect(screen.queryByText("Old result")).not.toBeInTheDocument();
  });

  it("confirms and deletes a completed generation before returning to create", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.endsWith("/jobs/job_delete") && init?.method === "DELETE") return Promise.resolve(new Response(null, { status: 204 }));
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);

    render(
      <MemoryRouter initialEntries={[{
        pathname: "/projects/project_delete",
        search: "?job=job_delete",
        state: {
          job: {
            id: "job_delete",
            projectId: "project_delete",
            status: "completed",
            result: { clips: [{ id: "clip_delete", title: "Delete me", start: 0, end: 10, duration: 10, assets: { videoUrl: "/delete.mp4" } }] },
          },
        },
      }]}>
        <Routes>
          <Route path="/projects/:projectId" element={<CreatePage />} />
          <Route path="/create" element={<div>Fresh create route</div>} />
        </Routes>
      </MemoryRouter>,
    );

    const deleteButton = await screen.findByRole("button", { name: /delete generation/i });
    await user.click(deleteButton);
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/cannot be undone/i));
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false);

    confirm.mockReturnValue(true);
    await user.click(deleteButton);
    expect(await screen.findByText("Fresh create route")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input, init]) => String(input).endsWith("/jobs/job_delete") && init?.method === "DELETE")).toBe(true);
  });

  it("switches generations and keeps the project open after deleting the current one", async () => {
    const user = userEvent.setup();
    const newest = {
      id: "job_newest",
      projectId: "project_history",
      status: "completed",
      createdAt: "2026-08-13T12:00:00Z",
      result: { clips: [{ id: "new_clip", title: "Newest result", start: 0, end: 10, duration: 10, assets: { videoUrl: "/new.mp4" } }] },
    };
    const older = {
      id: "job_older",
      projectId: "project_history",
      status: "completed",
      createdAt: "2026-08-12T12:00:00Z",
      result: { clips: [{ id: "old_clip", title: "Older result", start: 0, end: 10, duration: 10, assets: { videoUrl: "/old.mp4" } }] },
    };
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const requestUrl = String(input);
      if (requestUrl.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex", capabilities: {} });
      if (requestUrl.endsWith("/provider-profiles")) return response([{ id: "profile_codex", providerId: "codex", name: "Codex", model: "codex/default", active: true, generationReady: true, credential: { configured: true } }]);
      if (requestUrl.endsWith("/projects/project_history")) return response({
        id: "project_history",
        createdAt: "2026-08-12T11:00:00Z",
        source: { kind: "youtube", url: "https://www.youtube.com/watch?v=history0001" },
        jobs: [newest, older],
      });
      if (requestUrl.endsWith("/jobs/job_newest") && init?.method === "DELETE") {
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <MemoryRouter initialEntries={[{
        pathname: "/projects/project_history",
        search: "?job=job_newest",
        state: { job: newest },
      }]}>
        <Routes><Route path="/projects/:projectId" element={<CreatePage />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("combobox", { name: /viewing generation/i })).toHaveValue("job_newest");
    await user.click(screen.getByRole("button", { name: /delete generation/i }));
    expect(await screen.findByText("Older result")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /viewing generation/i })).toHaveValue("job_older");
    expect(screen.queryByText("Newest result")).not.toBeInTheDocument();
  });
});
