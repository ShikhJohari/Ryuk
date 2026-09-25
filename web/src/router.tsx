import type { QueryClient } from "@tanstack/react-query";
import { createRouter, type RouterHistory } from "@tanstack/react-router";
import { RouteError } from "@/components/route-error";
import { routeTree } from "./routeTree.gen";

type CreateAppRouterOptions = {
  readonly queryClient: QueryClient;
  /** Defaults to browser history; tests pass a memory history. */
  readonly history?: RouterHistory;
};

export function createAppRouter({
  queryClient,
  history,
}: CreateAppRouterOptions) {
  return createRouter({
    routeTree,
    context: { queryClient },
    ...(history === undefined ? {} : { history }),
    defaultPreload: "intent",
    // TanStack Query owns caching; the router always asks it.
    defaultPreloadStaleTime: 0,
    defaultErrorComponent: RouteError,
    scrollRestoration: true,
  });
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof createAppRouter>;
  }
}
