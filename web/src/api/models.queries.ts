import { queryOptions } from "@tanstack/react-query";
import { runQuery } from "@/lib/runtime";
import { listModels } from "./models";

export const modelsKey = ["models"] as const;

export const modelsQueryOptions = queryOptions({
  queryKey: modelsKey,
  queryFn: ({ signal }) => runQuery(listModels, { signal }),
});
