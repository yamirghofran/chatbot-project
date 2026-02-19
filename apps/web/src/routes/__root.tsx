import { Outlet, createRootRoute, useRouterState } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: Root,
});

function Root() {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  const isHomePage = pathname === "/";

  return (
    <div className="min-h-dvh">
      <main className={isHomePage ? "px-4 py-8" : "mx-auto max-w-5xl px-4 py-8"}>
        <Outlet />
      </main>
    </div>
  );
}
