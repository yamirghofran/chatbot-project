import { Outlet, createRootRoute } from "@tanstack/react-router";
import { Navbar } from "@/components/navigation/Navbar";

export const Route = createRootRoute({
  component: Root,
});

function Root() {
  return (
    <div className="min-h-dvh">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
