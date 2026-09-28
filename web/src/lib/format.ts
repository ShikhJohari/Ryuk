const dateFormat = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium" });

/**
 * A service timestamp as the operator reads it, such as "27 Sept 2026". One
 * that cannot be read is shown as sent, rather than failing the whole page.
 */
export function formatDate(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : dateFormat.format(date);
}

/** A match score or threshold, as the operator reads it: 0.874. */
export function formatScore(score: number): string {
  return score.toFixed(3);
}
