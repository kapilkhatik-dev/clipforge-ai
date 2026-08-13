import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import type { FormEvent } from "react";
import { describe, expect, it, vi } from "vitest";
import { Button, SelectMenu } from "./Ui";

describe("Button", () => {
  it("does not submit an enclosing form unless submit is explicitly requested", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event: FormEvent) => event.preventDefault());

    render(
      <form onSubmit={onSubmit}>
        <Button>Preview</Button>
        <Button type="submit">Generate</Button>
      </form>,
    );

    await user.click(screen.getByRole("button", { name: "Preview" }));
    expect(onSubmit).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect(onSubmit).toHaveBeenCalledOnce();
  });
});

function SelectHarness() {
  const [value, setValue] = useState("auto");
  return (
    <div>
      <label htmlFor="creative-direction">Creative direction</label>
      <SelectMenu
        id="creative-direction"
        ariaLabel="Creative direction"
        value={value}
        onChange={setValue}
        options={[
          { value: "auto", label: "Auto-detect video type" },
          { value: "comedy", label: "Comedy" },
          { value: "podcast", label: "Podcast" },
        ]}
      />
      <output aria-label="Selected direction">{value}</output>
    </div>
  );
}

describe("SelectMenu", () => {
  it("opens an aligned listbox and selects an option with the keyboard", async () => {
    const user = userEvent.setup();
    render(<SelectHarness />);

    const combobox = screen.getByRole("combobox", { name: "Creative direction" });
    expect(combobox).toHaveAttribute("aria-expanded", "false");

    combobox.focus();
    await user.keyboard("{ArrowDown}");
    expect(combobox).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("listbox")).toBeVisible();

    await user.keyboard("{Enter}");
    expect(combobox).toHaveTextContent("Comedy");
    expect(screen.getByRole("status", { name: "Selected direction" })).toHaveTextContent("comedy");
    expect(combobox).toHaveFocus();
    expect(combobox).toHaveAttribute("aria-expanded", "false");
  });

  it("closes on Tab while preserving normal focus navigation", async () => {
    const user = userEvent.setup();
    render(<SelectHarness />);

    const combobox = screen.getByRole("combobox", { name: "Creative direction" });
    await user.click(combobox);
    await user.tab();

    expect(combobox).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(combobox).not.toHaveFocus();
  });

  it("closes without changing the value when Escape is pressed", async () => {
    const user = userEvent.setup();
    render(<SelectHarness />);

    const combobox = screen.getByRole("combobox", { name: "Creative direction" });
    await user.click(combobox);
    await user.keyboard("{ArrowDown}{Escape}");

    expect(combobox).toHaveTextContent("Auto-detect video type");
    expect(combobox).toHaveAttribute("aria-expanded", "false");
  });

  it("finds options by typing their label and confirms the active match", async () => {
    const user = userEvent.setup();
    render(<SelectHarness />);

    const combobox = screen.getByRole("combobox", { name: "Creative direction" });
    combobox.focus();
    await user.keyboard("po");

    expect(combobox).toHaveAttribute("aria-expanded", "true");
    const podcast = screen.getByRole("option", { name: "Podcast" });
    expect(combobox).toHaveAttribute("aria-activedescendant", podcast.id);

    await user.keyboard("{Enter}");
    expect(combobox).toHaveTextContent("Podcast");
    expect(screen.getByRole("status", { name: "Selected direction" })).toHaveTextContent("podcast");
  });
});
