import { queryOptions } from "@tanstack/react-query";
import { runQuery } from "@/lib/runtime";
import { getHealth } from "./health";

export const healthQueryOptions = queryOptions({
  queryKey: ["health"],
  queryFn: ({ signal }) => runQuery(getHealth, { signal }),
  // Polled, so a failed check is simply retried on the next tick; retrying
  // immediately would only delay showing that the service is unreachable.
  refetchInterval: 15_000,
  retry: false,
});
