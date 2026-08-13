import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const response = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), {
  status: 200,
  headers: { "Content-Type": "application/json" },
}));

describe("App routes", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/bootstrap")) return response({ defaultProviderProfileId: "profile_codex" });
      if (url.endsWith("/provider-profiles")) return response([]);
      return response({});
    }));
  });

  it("does not expose the retired settings page", () => {
    render(
      <MemoryRouter initialEntries={["/settings/providers"]}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByRole("heading", { name: /that frame/i })).toBeInTheDocument();
    expect(screen.queryByText(/one workspace, any intelligence/i)).not.toBeInTheDocument();
  });
});
