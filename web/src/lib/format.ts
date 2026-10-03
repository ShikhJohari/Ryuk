const dateFormat = new Intl.DateTimeFormat("en-GB", { dateStyle: "medium" });
const dateTimeFormat = new Intl.DateTimeFormat("en-GB", {
  dateStyle: "medium",
  timeStyle: "medium",
});
const timeFormat = new Intl.DateTimeFormat("en-GB", { timeStyle: "medium" });

/**
 * A service timestamp in `format`, in the operator's local time. One that
 * cannot be read is shown as sent, rather than failing the whole page.
 */
function formatTimestamp(
  format: Intl.DateTimeFormat,
  timestamp: string,
): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : format.format(date);
}

/** A service timestamp as the operator reads it, such as "27 Sept 2026". */
export function formatDate(timestamp: string): string {
  return formatTimestamp(dateFormat, timestamp);
}

/**
 * A service timestamp to the second, such as "27 Sept 2026, 10:00:05": a
 * sighting's span is seconds long.
 */
export function formatDateTime(timestamp: string): string {
  return formatTimestamp(dateTimeFormat, timestamp);
}

/** A service timestamp's time of day to the second, such as "10:00:05". */
export function formatTime(timestamp: string): string {
  return formatTimestamp(timeFormat, timestamp);
}

/** A match score or threshold, as the operator reads it: 0.874. */
export function formatScore(score: number): string {
  return score.toFixed(3);
}
