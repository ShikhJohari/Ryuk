import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { NotFound } from "@/components/not-found";
import { SiteHeader } from "@/components/site-header";

export type RouterContext = {
  readonly queryClient: QueryClient;
};

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
  notFoundComponent: NotFound,
});

function RootLayout() {
  return (
    <>
      <SiteHeader />
      <main className="px-14 py-8">
        <Outlet />
      </main>
    </>
  );
}
