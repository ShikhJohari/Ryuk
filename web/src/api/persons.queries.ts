import { queryOptions } from "@tanstack/react-query";
import { runQuery } from "@/lib/runtime";
import { getPerson, listPersons, type StatusFilter } from "./persons";

/** Every watchlist query starts with this key, so one invalidation refreshes them all. */
export const personsKey = ["persons"] as const;

export const personsQueryOptions = (status: StatusFilter) =>
  queryOptions({
    queryKey: [...personsKey, "list", status],
    queryFn: ({ signal }) => runQuery(listPersons(status), { signal }),
  });

export const personQueryOptions = (personId: string) =>
  queryOptions({
    queryKey: [...personsKey, "detail", personId],
    queryFn: ({ signal }) => runQuery(getPerson(personId), { signal }),
  });
