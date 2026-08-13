import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { EditorPlaceholderPage } from "./EditorPlaceholderPage";

describe("EditorPlaceholderPage", () => {
  it("preserves a safe return path to generated results", () => {
    render(
      <MemoryRouter initialEntries={["/projects/project_1/clips/clip_1/edit?returnTo=%2Fprojects%2Fproject_1%3Fjob%3Djob_1"]}>
        <Routes>
          <Route path="/projects/:projectId/clips/:clipId/edit" element={<EditorPlaceholderPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByRole("link", { name: /back to results/i })).toHaveAttribute(
      "href",
      "/projects/project_1?job=job_1",
    );
  });
});
