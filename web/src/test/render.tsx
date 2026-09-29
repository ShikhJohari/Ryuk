import { QueryClient } from "@tanstack/react-query";
import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { render } from "@testing-library/react";
import { type ReactNode, StrictMode } from "react";
import { App } from "@/app";
import { createAppRouter } from "@/router";

/**
 * The whole app at `path`, against whatever the test's mock service answers,
 * under StrictMode as in main.tsx: effects mount, unmount and mount again.
 */
export function renderAt(path: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const router = createAppRouter({
    queryClient,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(
    <StrictMode>
      <App queryClient={queryClient} router={router} />
    </StrictMode>,
  );
  return router;
}

/**
 * `ui` alone, under a router of its own so its links render: a link's
 * `href` is built as in the app, though nothing it leads to is there. The
 * router renders it asynchronously, so find what it shows with `findBy`.
 */
export function renderWithRouter(ui: ReactNode) {
  const router = createRouter({
    routeTree: createRootRoute({ component: () => ui }),
    history: createMemoryHistory({ initialEntries: ["/"] }),
  });
  return render(
    <StrictMode>
      <RouterProvider router={router} />
    </StrictMode>,
  );
}
