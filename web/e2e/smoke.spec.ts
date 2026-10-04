import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { CAMERA_PICTURE } from "../playwright.config";

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

test("boxes the face of a person just enrolled on the live monitor", async ({
  page,
}) => {
  // Chrome's fake camera shows the same picture as the one enrolled.
  await page.goto("/watchlist");
  await page.getByRole("button", { name: "Enroll" }).click();
  const dialog = page.getByRole("dialog", {
    name: "Enroll a person of interest",
  });
  await dialog.getByLabel("Name").fill("Eileen Collins");
  await dialog.getByLabel("Photo").setInputFiles({
    name: "eileen-collins.jpg",
    mimeType: "image/jpeg",
    buffer: readFileSync(CAMERA_PICTURE),
  });
  await dialog.getByRole("button", { name: "Enroll" }).click();
  await expect(
    page.getByRole("heading", { level: 1, name: "Eileen Collins" }),
  ).toBeVisible();

  await page.getByRole("link", { name: "Live monitor" }).click();

  await expect(page.getByText("SFace", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("img", { name: /^Match: Eileen Collins, score / }),
  ).toBeVisible();
});

test("shows the committed evaluation as numbered tables and figures", async ({
  page,
}) => {
  await page.goto("/evaluation");

  await expect(
    page.getByRole("heading", { level: 1, name: "Evaluation" }),
  ).toBeVisible();
  await expect(
    page.getByRole("table", { name: /^Table 1\. Recognition models\./ }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: /^Figure 1\. TAR against FAR on LFW View 2\./,
    }),
  ).toBeVisible();
});
