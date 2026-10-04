import { queryOptions } from "@tanstack/react-query";
import { runQuery } from "@/lib/runtime";
import { getEvaluation } from "./evaluation";

export const evaluationKey = ["evaluation"] as const;

export const evaluationQueryOptions = queryOptions({
  queryKey: evaluationKey,
  queryFn: ({ signal }) => runQuery(getEvaluation, { signal }),
  // The service reads the committed results once, at startup: only a restart
  // changes them, and a reload picks that up.
  staleTime: Number.POSITIVE_INFINITY,
});
