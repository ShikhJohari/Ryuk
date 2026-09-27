import { expect, test } from "@playwright/test";

test("opens on the Live monitor with the service connected", async ({
  page,
}) => {
  await page.goto("/");

  await expect(page).toHaveURL(/\/monitor$/);
  await expect(
    page.getByRole("heading", { level: 1, name: "Live monitor" }),
  ).toBeVisible();
  await expect(page.getByText("Service connected")).toBeVisible();
});

test("shows the watchlist from the service", async ({ page }) => {
  await page.goto("/watchlist");

  await expect(
    page.getByRole("heading", { level: 1, name: "Watchlist" }),
  ).toBeVisible();
  await expect(
    page.getByText("Nobody is on the watchlist yet. Enroll someone to start."),
  ).toBeVisible();
});
