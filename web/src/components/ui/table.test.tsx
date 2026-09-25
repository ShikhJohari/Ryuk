import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./table";

describe("Table", () => {
  it("renders an accessible ruled table", () => {
    render(
      <Table>
        <TableCaption>Persons of interest on the watchlist.</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Enrolled photos</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell>Ada Lovelace</TableCell>
            <TableCell>3</TableCell>
          </TableRow>
        </TableBody>
      </Table>,
    );

    const table = screen.getByRole("table", {
      name: "Persons of interest on the watchlist.",
    });
    expect(table).toHaveClass("border-y-2", "border-ink");
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((th) => th.textContent),
    ).toEqual(["Name", "Enrolled photos"]);
    expect(
      within(table).getByRole("cell", { name: "Ada Lovelace" }),
    ).toBeInTheDocument();
  });
});
