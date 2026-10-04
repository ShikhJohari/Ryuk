import { screen, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";
import type { EvaluationReport } from "@/api/evaluation";
import { healthy, mockService } from "./test/api-server";
import {
  evaluationHandler,
  evaluationReport,
  evaluationUnavailable,
} from "./test/evaluation";
import { arcface, facenet, sface } from "./test/monitor";
import { renderAt } from "./test/render";

const server = mockService(
  healthy,
  http.get("*/api/models", () => HttpResponse.json([sface, facenet, arcface])),
  evaluationHandler(),
);

/** The page, once its heading shows. */
async function renderEvaluation() {
  renderAt("/evaluation");
  await screen.findByRole("heading", { level: 1, name: "Evaluation" });
  return screen.getByRole("main");
}

function region(name: string) {
  return screen.getByRole("region", { name });
}

function table(name: RegExp) {
  return screen.getByRole("table", { name });
}

function figure(name: RegExp) {
  return screen.getByRole("img", { name });
}

/** A table's rows as the text of their cells, header row first. */
function rowsOf(element: HTMLElement) {
  return within(element)
    .getAllByRole("row")
    .map((row) =>
      Array.from(row.querySelectorAll("th, td")).map(
        (cell) => cell.textContent,
      ),
    );
}

/** The row whose first cell reads `first`. */
function rowOf(element: HTMLElement, first: string) {
  const row = rowsOf(element).find((cells) => cells[0] === first);
  if (row === undefined) {
    throw new Error(`no row ${first}`);
  }
  return row;
}

/** The series an SVG chart draws, by network, with its dash pattern. */
function seriesOf(svg: HTMLElement) {
  return Array.from(svg.querySelectorAll("[data-series]")).map((path) => ({
    network: path.getAttribute("data-series"),
    stroke: path.getAttribute("stroke"),
    dashes: path.getAttribute("stroke-dasharray"),
  }));
}

function withChanges(changes: Partial<EvaluationReport>): EvaluationReport {
  return { ...evaluationReport, ...changes };
}

describe("evaluation page", () => {
  it("shows every section in the report's order", async () => {
    const main = await renderEvaluation();

    expect(
      within(main)
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual([
      "Recognition models",
      "Datasets",
      "Verification on LFW",
      "Watchlist search on CelebA",
      "Learning on embeddings",
      "Bias breakdown",
      "Away from the rehearsal",
    ]);
  });

  it("numbers every table and figure in page order, each named by its caption", async () => {
    const main = await renderEvaluation();

    const tables = within(main)
      .getAllByRole("table")
      .map((element) => element.querySelector("caption")?.textContent ?? "");
    expect(
      tables.map((caption) => caption.split(" ").slice(0, 2).join(" ")),
    ).toEqual(Array.from({ length: 14 }, (_, index) => `Table ${index + 1}.`));
    expect(tables.map((caption) => caption.split(".")[1]?.trim())).toEqual([
      "Recognition models",
      "Eligibility for the first active model",
      "The datasets, from images to usable faces",
      "LFW's pairs lists",
      "Verification on LFW View 2",
      "Watchlist search on the CelebA test draw",
      "Learning on embeddings for SFace",
      "Learning on embeddings for ArcFace (CPU)",
      "Learning on embeddings for FaceNet",
      "Rates by group for SFace under best photo",
      "Rates by group for ArcFace (CPU) under best photo",
      "Rates by group for FaceNet under best photo",
      "The live rules on smaller watchlists",
      "The same-person warning",
    ]);

    const figures = within(main)
      .getAllByRole("img")
      .filter((element) => element.tagName.toLowerCase() === "svg");
    expect(figures.map((svg) => svg.getAttribute("aria-labelledby"))).toEqual(
      figures.map((svg) => {
        const caption = svg.closest("figure")?.querySelector("figcaption");
        return caption?.id;
      }),
    );
    expect(
      figures.map((svg) => {
        const name = svg.closest("figure")?.querySelector("figcaption");
        return name?.textContent?.split(".").slice(0, 2).join(".");
      }),
    ).toEqual([
      "Figure 1. TAR against FAR on LFW View 2",
      "Figure 2. TPIR against FPIR on the CelebA test draw",
      "Figure 3. Each group's test FPIR at the model's single frozen threshold under best photo, with its 95% identity-level interval",
    ]);
  });

  describe("recognition models", () => {
    it("lists every model read-only, its ms per face end to end", async () => {
      const main = await renderEvaluation();

      const models = table(/^Table 1\. Recognition models/);
      expect(rowsOf(models)).toEqual([
        [
          "Model",
          "State",
          "Embedding size",
          "Threshold",
          "ms per face (end to end)",
        ],
        ["SFace", "Active", "128", "0.498", "5.4"],
        ["FaceNet", "Available", "128", "0.709", "5.4"],
        ["ArcFace (CPU)", "Not evaluated", "128", "—", "—"],
      ]);
      expect(models).toHaveAccessibleName(
        expect.stringContaining(
          "from pixels to top candidate, detection included",
        ),
      );
      // #20: switching models happens only in the live monitor's toolbar.
      expect(within(main).queryAllByRole("combobox")).toEqual([]);
      expect(within(main).queryAllByRole("button")).toEqual([]);
      expect(within(main).queryByText(/switch model/i)).toBeNull();
      expect(
        within(region("Recognition models")).getByRole("link", {
          name: "live monitor",
        }),
      ).toHaveAttribute("href", "/monitor");
    });

    it("names the first active model, its reason, and how each model was judged", async () => {
      await renderEvaluation();

      const models = region("Recognition models");
      expect(
        within(models).getByText(
          "ArcFace (CPU) has the highest test TPIR at its frozen threshold (98.47%) of the 2 eligible models.",
        ),
      ).toBeInTheDocument();
      expect(
        within(models).getByText("First active model").nextElementSibling,
      ).toHaveTextContent("ArcFace (CPU)");
      const eligibility = table(
        /^Table 2\. Eligibility for the first active model/,
      );
      expect(eligibility).toHaveAccessibleName(
        expect.stringContaining(
          "within 0.5 points of its published accuracy over the 5,917 scored pairs of 6,000, test FPIR at the frozen threshold at most 2%, and at most 30 ms per face end to end",
        ),
      );
      expect(rowsOf(eligibility)).toEqual([
        [
          "Model",
          "LFW gap (points)",
          "Test TPIR",
          "Test FPIR",
          "ms per face (end to end)",
          "Eligible",
        ],
        ["SFace", "−0.02", "96.2 [95.4–96.9]", "0.74", "5.4", "Yes"],
        ["ArcFace (CPU)", "−0.05", "98.5 [98.1–98.8]", "0.66", "8.3", "Yes"],
        [
          "FaceNet",
          "−0.29",
          "88.4 [87.0–89.7]",
          "2.51 over the limit",
          "11.7",
          "No",
        ],
      ]);
    });
  });

  it("counts each dataset from images to usable faces, and LFW's pairs lists", async () => {
    await renderEvaluation();

    const datasets = table(/^Table 3\. The datasets/);
    expect(datasets).toHaveAccessibleName(
      expect.stringContaining(
        "at least 70 px on the detection box's short side",
      ),
    );
    expect(rowsOf(datasets)).toEqual([
      [
        "Dataset",
        "Identities",
        "Images",
        "Face found",
        "More than one face",
        "Usable face",
        "Gallery candidates",
        "Eligible identities",
      ],
      [
        "LFW",
        "5,749",
        "13,233",
        "13,161 (99.5%)",
        "654 (4.9%)",
        "13,144 (99.3%)",
        "—",
        "—",
      ],
      [
        "CelebA validation draw (valid split)",
        "985",
        "19,867",
        "19,175 (96.5%)",
        "7 (0.0%)",
        "19,070 (96.0%)",
        "629",
        "584",
      ],
      [
        "CelebA test draw (test split)",
        "1,000",
        "19,962",
        "19,314 (96.8%)",
        "20 (0.1%)",
        "19,228 (96.3%)",
        "616",
        "572",
      ],
    ]);

    expect(rowsOf(table(/^Table 4\. LFW's pairs lists/))).toEqual([
      [
        "List",
        "View",
        "Folds",
        "Matched",
        "Mismatched",
        "Identities",
        "Images",
        "Pairs not scored",
      ],
      ["pairsDevTrain", "1", "1", "1,100", "1,100", "2,132", "3,443", "32"],
      ["pairsDevTest", "1", "1", "500", "500", "963", "1,549", "16"],
      ["pairs", "2", "10", "3,000", "3,000", "4,281", "7,701", "83"],
    ]);
  });

  describe("verification on LFW", () => {
    it("gives both accuracies, and marks TAR at FAR 0.1% indicative", async () => {
      await renderEvaluation();

      const lfw = table(/^Table 5\. Verification on LFW View 2/);
      expect(lfw).toHaveAccessibleName(
        expect.stringContaining(
          "5,917 of 6,000 pairs scored, in 10 folds, the threshold of each fold chosen on the other nine",
        ),
      );
      expect(rowsOf(lfw)).toEqual([
        [
          "Model",
          "Embedding size",
          "Published LFW",
          "Our LFW (± SE)",
          "Unscored pairs as errors",
          "AUC",
          "TAR @ FAR 1%",
          "TAR @ FAR 0.1% (indicative)",
        ],
        [
          "SFace",
          "128",
          "99.40",
          "99.38 ± 0.12",
          "98.00",
          "0.9977",
          "99.33",
          "98.35",
        ],
        [
          "ArcFace (CPU)",
          "512",
          "99.83",
          "99.78 ± 0.12",
          "98.40",
          "0.9977",
          "99.70",
          "99.63",
        ],
        [
          "FaceNet",
          "512",
          "99.65",
          "99.36 ± 0.12",
          "97.98",
          "0.9977",
          "99.36",
          "97.94",
        ],
      ]);
    });

    it("says FaceNet's published ± is a fold standard deviation and the int8 figure times the embedding alone", async () => {
      await renderEvaluation();

      const lfw = region("Verification on LFW");
      expect(
        within(lfw).getByText(
          /^FaceNet's published figure: 99\.65 ± 0\.25, measured on MTCNN crops with a margin of 32\. Its ± is a standard deviation across the source's folds, not a standard error like ours\.$/,
        ),
      ).toBeInTheDocument();
      expect(
        within(lfw).getByText(
          /^ArcFace \(CPU\)'s published figure: The buffalo_l pack's figure\.$/,
        ),
      ).toBeInTheDocument();
      expect(
        within(lfw).getByText(
          /^SFace int8 is not a row: Ryuk runs SFace fp32\. On the same pairs int8 scores 99\.12 ± 0\.13, and takes 11\.4 ms per face against fp32's 4\.1 ms, timing the embedding alone, not end to end like every other ms per face on this page\./,
        ),
      ).toHaveTextContent(
        "Its embeddings also drift from fp32's: cosine 0.971 on average, 0.930 at worst, over 7,643 faces.",
      );
      expect(
        within(lfw).getByText(
          /each of the 83 unscored pairs counted as an error/,
        ),
      ).toBeInTheDocument();
      expect(within(lfw).queryByText(/⚑/)).toBeNull();
    });

    it("flags a model that does not reproduce its published accuracy", async () => {
      const [sfaceResult, ...others] = evaluationReport.verification.models;
      if (sfaceResult === undefined) {
        throw new Error("the fixture has no models");
      }
      server.use(
        evaluationHandler(
          withChanges({
            verification: {
              ...evaluationReport.verification,
              models: [
                { ...sfaceResult, reproducesPublished: false },
                ...others,
              ],
            },
          }),
        ),
      );
      await renderEvaluation();

      expect(rowOf(table(/^Table 5\./), "SFace")[3]).toBe("99.38 ± 0.12 ⚑");
      expect(
        within(region("Verification on LFW")).getByText(
          "⚑ More than 0.5 points from the published figure: a pipeline bug to find, not a result.",
        ),
      ).toBeInTheDocument();
    });

    it("draws each model's ROC in its own colour and dash, labelled directly, with its operating points", async () => {
      await renderEvaluation();

      const roc = figure(/^Figure 1\. TAR against FAR on LFW View 2/);
      expect(roc.tagName.toLowerCase()).toBe("svg");
      expect(seriesOf(roc)).toEqual([
        {
          network: "sface",
          stroke: "var(--color-model-sface)",
          dashes: "1.92 2.88",
        },
        {
          network: "arcface",
          stroke: "var(--color-model-arcface)",
          dashes: null,
        },
        {
          network: "facenet",
          stroke: "var(--color-model-facenet)",
          dashes: "11.2 4",
        },
      ]);
      // Each model's FAR 0.1% point is indicative, and drawn hollow so the chart marks it.
      const markers = Array.from(roc.querySelectorAll("[data-marker]"));
      expect(markers).toHaveLength(6);
      const hollow = markers.filter((marker) =>
        marker.hasAttribute("data-indicative"),
      );
      expect(hollow).toHaveLength(3);
      for (const marker of hollow) {
        expect(marker).toHaveAttribute("fill", "var(--color-paper)");
      }
      for (const label of [
        "SFace, 99.38%",
        "ArcFace (CPU), 99.78%",
        "FaceNet, 99.36%",
      ]) {
        expect(within(roc).getByText(label)).toBeInTheDocument();
      }
      expect(
        within(roc).getByText("False accept rate (log scale)"),
      ).toBeInTheDocument();
      for (const tick of ["0.01%", "0.1%", "1%", "10%"]) {
        expect(within(roc).getByText(tick)).toBeInTheDocument();
      }

      const readout = screen.getByRole("img", {
        name: "SFace: TAR 98.35% at FAR 0.068%, the operating point for FAR 0.1% (indicative)",
      });
      expect(readout).toHaveAttribute("tabindex", "0");
      expect(readout).toHaveTextContent(
        "SFace: TAR 98.35% at FAR 0.068%, the operating point for FAR 0.1% (indicative)",
      );
      expect(
        screen.getByRole("img", {
          name: "ArcFace (CPU): TAR 99.70% at FAR 0.98%, the operating point for FAR 1%",
        }),
      ).toBeInTheDocument();
    });
  });

  describe("watchlist search on CelebA", () => {
    it("gives each model's rates at its frozen threshold, its curve's indicative point, and its ms per face end to end", async () => {
      await renderEvaluation();

      const openSet = table(
        /^Table 6\. Watchlist search on the CelebA test draw/,
      );
      expect(rowsOf(openSet)).toEqual([
        ["Measure", "SFace", "ArcFace (CPU) ★", "FaceNet"],
        ["Rank-1", "98.0 [97.6–98.5]", "98.0 [97.6–98.5]", "98.0 [97.6–98.5]"],
        ["Frozen threshold", "0.498", "0.342", "0.709"],
        ["TPIR", "95.0 [94.2–95.8]", "98.5 [97.6–99.2]", "88.4 [87.5–89.1]"],
        [
          "FPIR",
          "0.91 [0.62–1.24]Wilson [0.64–1.29]",
          "0.66 [0.62–1.24]Wilson [0.64–1.29]",
          "0.96 [0.62–1.24]Wilson [0.64–1.29]",
        ],
        [
          "Misidentification",
          "0.16 [0.05–0.31]",
          "0.16 [0.05–0.31]",
          "0.16 [0.05–0.31]",
        ],
        [
          "TPIR @ FPIR 1%",
          "95.2 [94.3–96.0]",
          "98.7 [97.8–99.5]",
          "88.6 [87.7–89.4]",
        ],
        [
          "TPIR @ FPIR 0.1% (indicative)",
          "90.0 [88.5–92.6]",
          "93.5 [92.0–96.1]",
          "83.4 [81.9–86.0]",
        ],
        ["ms per face (end to end)", "5.4", "8.3", "11.7"],
      ]);
      expect(openSet).toHaveAccessibleName(
        expect.stringContaining("frozen on the validation draw at FPIR 1%"),
      );
      const celeba = region("Watchlist search on CelebA");
      expect(
        within(celeba).getByText(
          /^CelebA test draw: 500 gallery identities with 7,500 mated probes, and 500 held-out identities with 4,072 non-mated probes\./,
        ),
      ).toBeInTheDocument();
      expect(
        within(celeba).getByText(
          /^ms per face runs from pixels to top candidate, detection included/,
        ),
      ).toBeInTheDocument();
    });

    it("draws TPIR against FPIR from one false alarm, each frozen threshold marked", async () => {
      await renderEvaluation();

      const curves = figure(
        /^Figure 2\. TPIR against FPIR on the CelebA test draw/,
      );
      expect(seriesOf(curves).map((series) => series.network)).toEqual([
        "sface",
        "arcface",
        "facenet",
      ]);
      expect(curves).toHaveAccessibleName(
        expect.stringContaining("one false alarm in 4,072 non-mated probes"),
      );
      for (const name of ["SFace", "ArcFace (CPU)", "FaceNet"]) {
        expect(within(curves).getByText(name)).toBeInTheDocument();
      }
      expect(
        screen.getByRole("img", {
          name: "SFace at its frozen threshold, 0.498: TPIR 95.0%, FPIR 0.91%",
        }),
      ).toHaveAttribute("tabindex", "0");
    });
  });

  it("compares the methods per model, with the live rule and its reason", async () => {
    await renderEvaluation();

    const sfaceMethods = table(/^Table 7\. Learning on embeddings for SFace/);
    expect(rowsOf(sfaceMethods)).toEqual([
      [
        "Method",
        "Hyperparameter",
        "Rank-1",
        "TPIR @ FPIR 1%",
        "Gain (points)",
        "Frozen threshold",
        "TPIR",
        "FPIR",
        "Misidentification",
      ],
      [
        "Best photo",
        "—",
        "98.1 [97.6–98.5]",
        "95.3 [94.4–96.1]",
        "—",
        "0.498",
        "95.3 [94.5–96.1]",
        "0.91 [0.62–1.24]",
        "0.16 [0.05–0.31]",
      ],
      [
        "Mean ★",
        "—",
        "98.1 [97.6–98.5]",
        "96.6 [95.7–97.4]",
        "+1.4 [+0.6, +2.0]",
        "0.494",
        "96.6 [95.8–97.4]",
        "0.91 [0.62–1.24]",
        "0.16 [0.05–0.31]",
      ],
      [
        "kNN †",
        "k = 5",
        "98.1 [97.6–98.5]",
        "91.3 [90.4–92.1]",
        "−3.9 [−5.2, −2.7]",
        "0.494",
        "91.3 [90.5–92.1]",
        "0.91 [0.62–1.24]",
        "0.16 [0.05–0.31]",
      ],
      [
        "Learned rule",
        "—",
        "98.1 [97.6–98.5]",
        "95.4 [94.5–96.2]",
        "+0.1 [−0.1, +0.5]",
        "0.494",
        "95.4 [94.6–96.2]",
        "0.91 [0.62–1.24]",
        "0.16 [0.05–0.31]",
      ],
    ]);
    // Only a gain whose interval lies wholly above zero is bold.
    const bold = Array.from(sfaceMethods.querySelectorAll("strong")).map(
      (element) => element.textContent,
    );
    expect(bold).toEqual(["+1.4 [+0.6, +2.0]"]);

    const learning = region("Learning on embeddings");
    expect(
      within(learning).getByText(
        "The mean rule improves test TPIR at FPIR 1% on best-photo by +1.36 points.",
      ).parentElement,
    ).toHaveTextContent(/^SFace runs the mean rule live\./);
    expect(
      within(learning).getByText(
        "The learned rule improves test TPIR at FPIR 1% the most.",
      ).parentElement,
    ).toHaveTextContent(/^FaceNet runs the learned rule live\./);
    expect(
      within(learning).getByText(
        "† Needs retraining whenever the watchlist changes, so never runs live, whatever its gain.",
      ),
    ).toBeInTheDocument();
    expect(rowOf(table(/^Table 9\./), "Learned rule ★")[4]).toBe(
      "+1.9 [+0.9, +2.7]",
    );
  });

  describe("bias breakdown", () => {
    it("draws each model's group FPIR under best photo only, against its overall FPIR", async () => {
      await renderEvaluation();

      const groups = figure(/^Figure 3\. /);
      // A panel per model under best photo; SFace's mean breakdown is left out.
      expect(
        Array.from(groups.querySelectorAll("[data-panel]")).map((panel) =>
          panel.getAttribute("data-panel"),
        ),
      ).toEqual(["SFace", "ArcFace (CPU)", "FaceNet"]);
      expect(groups.querySelectorAll("[data-overall]")).toHaveLength(3);
      expect(
        groups.querySelectorAll("[data-indicative]").length,
      ).toBeGreaterThan(0);
      expect(within(groups).getAllByText("too few (n = 23)")).toHaveLength(3);
      expect(within(groups).getByText("indicative")).toBeInTheDocument();
      expect(
        screen.getByRole("img", {
          name: "SFace, Not male: FPIR 1.38% [0.89–1.95]",
        }),
      ).toHaveAttribute("tabindex", "0");
      expect(
        screen.queryByRole("img", { name: /^SFace, Not male, not young/ }),
      ).toBeNull();
    });

    it("leaves out the overall line, and says why, when the CelebA draw is not measured", async () => {
      server.use(
        evaluationHandler(
          withChanges({ identification: null, firstActiveModel: null }),
        ),
      );
      await renderEvaluation();

      const groups = figure(
        / under best photo, with its 95% identity-level interval\./,
      );
      expect(groups.querySelectorAll("[data-panel]")).toHaveLength(3);
      expect(groups.querySelectorAll("[data-overall]")).toHaveLength(0);
      expect(groups).toHaveAccessibleName(
        expect.stringContaining(
          "Each model's overall test FPIR is not drawn: the CelebA test draw is not measured yet.",
        ),
      );
      expect(groups).not.toHaveAccessibleName(
        expect.stringContaining("The dashed line"),
      );
    });

    it("tables each model's rates by group, each attribute's basis labelled", async () => {
      await renderEvaluation();

      const sfaceGroups = table(
        /^Table 10\. Rates by group for SFace under best photo/,
      );
      expect(sfaceGroups).toHaveAccessibleName(
        expect.stringContaining("at its single frozen threshold, 0.498"),
      );
      expect(rowsOf(sfaceGroups)).toEqual([
        [
          "Group",
          "Gallery identities",
          "Mated probes",
          "TPIR",
          "Misidentification",
          "Held-out identities",
          "Non-mated probes",
          "FPIR",
        ],
        ["Male · by identity · worst-to-best FPIR 3.87×"],
        [
          "Male",
          "177",
          "2,655",
          "97.8 [96.8–98.7]",
          "0.11 [0.03–0.23]",
          "240",
          "1,956",
          "0.36 [0.11–0.68]",
        ],
        [
          "Not male",
          "323",
          "4,845",
          "93.5 [92.2–94.7]",
          "0.19 [0.06–0.35]",
          "253",
          "2,095",
          "1.38 [0.89–1.95]",
        ],
        ["Male and young · by identity · indicative · worst-to-best FPIR —"],
        [
          "Male, young",
          "112",
          "1,680",
          "98.0 [96.8–99.0]",
          "0.12 [0.02–0.26]",
          "123",
          "1,001",
          "0.57 [0.14–1.12]",
        ],
        [
          "Not male, not young",
          "29",
          "435",
          "too few",
          "too few",
          "23",
          "180",
          "too few",
        ],
        ["Eyeglasses · per photo · worst-to-best FPIR 1.64×"],
        [
          "Eyeglasses",
          "150",
          "396",
          "92.2 [89.3–94.8]",
          "0.51 [0.00–1.27]",
          "121",
          "348",
          "0.57 [0.00–1.44]",
        ],
        [
          "No eyeglasses",
          "496",
          "7,104",
          "95.2 [94.4–95.9]",
          "0.15 [0.06–0.27]",
          "484",
          "3,724",
          "0.94 [0.64–1.27]",
        ],
      ]);
      const bias = region("Bias breakdown");
      expect(
        within(bias).getByText(
          /^By identity: each identity goes by its majority label over all its images, where at least 80% of them agree;/,
        ),
      ).toBeInTheDocument();
      expect(
        within(bias).getByText(/^Per photo: each probe goes by its own label/),
      ).toBeInTheDocument();
      expect(
        within(bias).getByText(
          "Too few: fewer than 30 identities stand behind the rate, so it is not estimated.",
        ),
      ).toBeInTheDocument();
      expect(screen.queryByRole("table", { name: /under mean/ })).toBeNull();
    });
  });

  describe("away from the rehearsal", () => {
    it("gives the live rules on 500, 100, 20 and 5 identities with 5 photos or 1", async () => {
      await renderEvaluation();

      const galleries = table(
        /^Table 13\. The live rules on smaller watchlists/,
      );
      const rows = rowsOf(galleries);
      expect(rows[0]).toEqual([
        "Model",
        "Live rule",
        "Identities",
        "Galleries",
        "TPIR, 5 photos",
        "FPIR, 5 photos",
        "TPIR, 1 photo",
        "FPIR, 1 photo",
      ]);
      expect(rows.slice(1, 5)).toEqual([
        [
          "SFace",
          "Mean",
          "500",
          "1",
          "96.2 [95.5–96.9]",
          "0.74 [0.44–1.04]",
          "78.3 [77.1–79.5]",
          "0.28 [0.14–0.45]",
        ],
        [
          "100",
          "5",
          "96.2 [95.5–96.9]",
          "0.15 [0.09–0.21]",
          "78.3 [77.1–79.5]",
          "0.056 [0.028–0.090]",
        ],
        [
          "20",
          "25",
          "96.2 [95.5–96.9]",
          "0.030 [0.018–0.041]",
          "78.3 [77.1–79.5]",
          "0.011 [0.006–0.018]",
        ],
        [
          "5",
          "100",
          "96.2 [95.5–96.9]",
          "0.007 [0.004–0.010]",
          "78.3 [77.1–79.5]",
          "0.003 [0.001–0.004]",
        ],
      ]);
      expect(rows).toHaveLength(13);
    });

    it("gives each model's same-person threshold, the warnings it raises and its FAR", async () => {
      await renderEvaluation();

      const samePerson = table(/^Table 14\. The same-person warning/);
      expect(samePerson).toHaveAccessibleName(
        expect.stringContaining(
          "frozen at FAR 0.1% on the validation draw's 5,707,000 impostor pairs",
        ),
      );
      expect(rowsOf(samePerson)).toEqual([
        ["Measure", "SFace", "ArcFace (CPU)", "FaceNet"],
        ["Same-person threshold", "0.377", "0.216", "0.521"],
        [
          "1:N threshold",
          "0.494 (mean)",
          "0.342 (best photo)",
          "0.635 (learned rule)",
        ],
        [
          "Own photos warned, 1 photo",
          "7.64 [6.98–8.32]",
          "7.64 [6.98–8.32]",
          "7.64 [6.98–8.32]",
        ],
        [
          "Own photos warned, 5 photos",
          "1.65 [1.31–2.03]",
          "1.65 [1.31–2.03]",
          "1.65 [1.31–2.03]",
        ],
        [
          "At the 1:N threshold, 1 photo",
          "21.5 [20.5–22.6]",
          "5.28 [4.71–5.89]",
          "—",
        ],
        [
          "At the 1:N threshold, 5 photos",
          "4.64 [4.12–5.19]",
          "1.49 [1.18–1.83]",
          "—",
        ],
        [
          "FAR, 1 photo",
          "0.09 [0.08–0.11]Wilson [0.08–0.11]",
          "0.09 [0.08–0.11]Wilson [0.08–0.11]",
          "0.09 [0.08–0.11]Wilson [0.08–0.11]",
        ],
        [
          "FAR, 5 photos",
          "0.36 [0.32–0.39]",
          "0.36 [0.32–0.39]",
          "0.36 [0.32–0.39]",
        ],
      ]);
    });
  });

  it("marks a learning method's headline TPIR indicative rather than hiding it", async () => {
    const learning = evaluationReport.learning;
    if (learning === null) {
      throw new Error("the fixture measures learning");
    }
    server.use(
      evaluationHandler(
        withChanges({
          learning: {
            ...learning,
            models: learning.models.map((compared) => ({
              ...compared,
              methods: compared.methods.map((method) => ({
                ...method,
                operatingPoints: method.operatingPoints.map((point) => ({
                  ...point,
                  indicative: true,
                })),
              })),
            })),
          },
        }),
      ),
    );
    await renderEvaluation();

    const sfaceMethods = table(/^Table 7\. Learning on embeddings for SFace/);
    expect(rowOf(sfaceMethods, "Best photo")[3]).toBe(
      "95.3 [94.4–96.1] (indicative)",
    );
  });

  it("says a model whose LFW pairs were not scored is not scored, and nothing more", async () => {
    const active = evaluationReport.firstActiveModel;
    if (active === null) {
      throw new Error("the fixture has a first active model");
    }
    server.use(
      evaluationHandler(
        withChanges({
          firstActiveModel: {
            ...active,
            eligibility: active.eligibility.map((judged) =>
              judged.model.network === "facenet"
                ? { ...judged, lfwGapPoints: null, reproducesLfw: false }
                : judged,
            ),
          },
        }),
      ),
    );
    await renderEvaluation();

    const eligibility = table(
      /^Table 2\. Eligibility for the first active model/,
    );
    expect(rowOf(eligibility, "FaceNet")[1]).toBe("not scored");
  });

  it("says which sections are not measured yet, numbering what is", async () => {
    server.use(
      evaluationHandler(
        withChanges({
          identification: null,
          firstActiveModel: null,
          learning: null,
          bias: null,
          live: null,
        }),
      ),
    );
    const main = await renderEvaluation();

    for (const [name, command] of [
      ["Watchlist search on CelebA", "ryuk evaluate celeba"],
      ["Learning on embeddings", "ryuk evaluate learn"],
      ["Bias breakdown", "ryuk evaluate bias"],
      ["Away from the rehearsal", "ryuk evaluate live"],
    ] as const) {
      const section = region(name);
      expect(within(section).getByText(/^Not measured yet/)).toHaveTextContent(
        `Not measured yet: run ${command}.`,
      );
      expect(within(section).queryByRole("table")).toBeNull();
      expect(within(section).queryByRole("img")).toBeNull();
    }
    expect(
      within(region("Recognition models")).getByText(/^Not measured yet/),
    ).toHaveTextContent("Not measured yet: run ryuk evaluate celeba.");
    expect(
      within(main)
        .getAllByRole("table")
        .map(
          (element) =>
            element.querySelector("caption")?.textContent?.split(".")[0],
        ),
    ).toEqual(["Table 1", "Table 2", "Table 3", "Table 4"]);
    expect(table(/^Table 4\. Verification on LFW View 2/)).toBeInTheDocument();
    expect(figure(/^Figure 1\. TAR against FAR/)).toBeInTheDocument();
  });

  it("says why when the service has no evaluation to show", async () => {
    server.use(evaluationUnavailable);
    renderAt("/evaluation");

    expect(
      await screen.findByRole("heading", { level: 1, name: "Evaluation" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The service was started without the evaluation results, so there is nothing to show. Restart it with `ryuk serve` from the repository root.",
    );
    expect(
      screen.getByRole("button", { name: "Try again" }),
    ).toBeInTheDocument();
  });
});
