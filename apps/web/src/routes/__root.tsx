import { useState } from "react";
import {
  Outlet,
  createRootRoute,
  useNavigate,
  useRouterState,
} from "@tanstack/react-router";
import { Navbar } from "@/components/navigation/Navbar";
import { useCurrentUser } from "@/lib/auth";

export const Route = createRootRoute({
  component: Root,
});

function Root() {
  const { data: user } = useCurrentUser();
  const navigate = useNavigate();
  const [searchValue, setSearchValue] = useState("");

  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isChat = pathname === "/chat";

  function handleSearchSubmit(rawValue: string) {
    const query = rawValue.trim();
    if (!query) return;
    setSearchValue(query);
    navigate({
      to: "/search",
      search: { q: query },
    });
  }

  return (
    <div className="flex flex-col h-dvh bg-background text-foreground">
      {user && (
        <Navbar
          user={user}
          searchValue={searchValue}
          onSearchChange={setSearchValue}
          onSearchSubmit={handleSearchSubmit}
        />
      )}
      <main className={isChat ? "flex-1 min-h-0" : "flex-1 w-full mx-auto max-w-5xl px-4 py-8"}>
        <Outlet />
      </main>
    </div>
  );
}
