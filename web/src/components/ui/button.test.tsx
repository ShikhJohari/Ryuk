import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Button } from "./button";

describe("Button", () => {
  it("is a non-submitting primary button by default", () => {
    render(<Button>Enroll</Button>);

    const button = screen.getByRole("button", { name: "Enroll" });
    expect(button).toHaveAttribute("type", "button");
    expect(button).toHaveAttribute("data-variant", "primary");
  });

  it("keeps an explicit type", () => {
    render(<Button type="submit">Save</Button>);

    expect(screen.getByRole("button", { name: "Save" })).toHaveAttribute(
      "type",
      "submit",
    );
  });
});
