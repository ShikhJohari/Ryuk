import { QueryClient } from "@tanstack/react-query";
import { createMemoryHistory } from "@tanstack/react-router";
import { render } from "@testing-library/react";
import { StrictMode } from "react";
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
