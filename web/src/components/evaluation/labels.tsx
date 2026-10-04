import type { BiasAttribute, MatchRule, Method, Rate } from "@/api/evaluation";
import { digitsFor, formatInterval, formatRate } from "./format";

/** The name a table gives each method, as `ryuk.evaluation.tables.METHOD_NAMES` does. */
export const METHOD_NAMES: Readonly<Record<Method, string>> = {
  "best-photo": "Best photo",
  mean: "Mean",
  knn: "kNN",
  "logistic-regression": "Logistic regression",
  "linear-svm": "Linear SVM",
  learned: "Learned rule",
};

/** A match rule as running text names it: "the mean rule". */
export const RULE_IN_PROSE: Readonly<Record<MatchRule, string>> = {
  "best-photo": "best photo",
  mean: "the mean rule",
  learned: "the learned rule",
};

/** The name a table gives each attribute of the bias breakdown. */
export const ATTRIBUTE_NAMES: Readonly<Record<BiasAttribute, string>> = {
  Male: "Male",
  Young: "Young",
  Male_and_Young: "Male and young",
  Eyeglasses: "Eyeglasses",
  Wearing_Hat: "Wearing a hat",
  Blurry: "Blurry",
};

/**
 * A rate and its bootstrap interval; under it, the dependence-adjusted
 * Wilson interval where evaluation computed one, for a rate within 1% of 0
 * or 100%.
 */
export function RateText({
  rate,
  digits = digitsFor(rate.value),
}: {
  readonly rate: Rate;
  readonly digits?: number;
}) {
  return (
    <>
      {formatRate(rate, digits)}
      {rate.adjustedWilson === null ? null : (
        <span className="block text-muted-foreground text-xs">
          Wilson {formatInterval(rate.adjustedWilson, digits)}
        </span>
      )}
    </>
  );
}
