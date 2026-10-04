import type { EvaluationReport } from "@/api/evaluation";
import { biasCounts } from "./bias-section";
import { DATASET_TABLES } from "./dataset-section";
import { identificationCounts } from "./identification-section";
import { learningTables } from "./learning-section";
import { LFW_FIGURES, LFW_TABLES } from "./lfw-section";
import { liveTables } from "./live-section";
import { modelsTables } from "./models-section";

/** A section's first table and figure numbers. */
export type Numbers = { readonly table: number; readonly figure: number };

export type SectionNumbers = {
  readonly models: Numbers;
  readonly dataset: Numbers;
  readonly lfw: Numbers;
  readonly identification: Numbers;
  readonly learning: Numbers;
  readonly bias: Numbers;
  readonly live: Numbers;
};

/**
 * Tables and figures are numbered in page order, so a section not measured
 * yet leaves no gap: each section starts where the one before it ended.
 */
export function sectionNumbers(report: EvaluationReport): SectionNumbers {
  const counts = [
    ["models", { tables: modelsTables(report.firstActiveModel), figures: 0 }],
    ["dataset", { tables: DATASET_TABLES, figures: 0 }],
    ["lfw", { tables: LFW_TABLES, figures: LFW_FIGURES }],
    ["identification", identificationCounts(report.identification)],
    ["learning", { tables: learningTables(report.learning), figures: 0 }],
    ["bias", biasCounts(report.bias)],
    ["live", { tables: liveTables(report.live), figures: 0 }],
  ] as const;
  let table = 1;
  let figure = 1;
  const numbers: Partial<Record<keyof SectionNumbers, Numbers>> = {};
  for (const [section, count] of counts) {
    numbers[section] = { table, figure };
    table += count.tables;
    figure += count.figures;
  }
  return numbers as SectionNumbers;
}
