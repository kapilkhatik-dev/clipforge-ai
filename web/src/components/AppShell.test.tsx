import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

const response = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), {
  status: 200,
  headers: { "Content-Type": "application/json" },
}));

describe("AppShell navigation", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex" });
      if (url.endsWith("/provider-profiles")) return response([{ id: "profile_codex", name: "Codex", active: true, generationReady: true }]);
      return response({});
    }));
  });

  it("keeps Create active throughout project routes", async () => {
    render(
      <MemoryRouter initialEntries={["/projects/project_1"]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="projects/:projectId" element={<div>Project result</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getAllByRole("link", { name: "Create" }).some((link) => link.getAttribute("aria-current") === "page")).toBe(true);
    expect(await screen.findByRole("status", { name: /local service online.*Codex/i })).toBeInTheDocument();
  });

  it("does not expose the removed settings navigation", () => {
    render(
      <MemoryRouter initialEntries={["/create"]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="create" element={<div>Create workspace</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByRole("link", { name: /providers|settings/i })).not.toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Create" })).toHaveLength(1);
  });
});
