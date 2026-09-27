const dateFormat = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium" });

/** A service timestamp as the operator reads it, such as "27 Sept 2026". */
export function formatDate(timestamp: string): string {
  return dateFormat.format(new Date(timestamp));
}

/** A match score or threshold, as the operator reads it: 0.874. */
export function formatScore(score: number): string {
  return score.toFixed(3);
}
