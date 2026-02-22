import { Link } from "@tanstack/react-router";
import type { User } from "@/lib/types";
import { SearchBar } from "@/components/search/SearchBar";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export type NavbarProps = {
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  onSearchSubmit?: (value: string) => void;
  brand?: string;
  user?: User;
};

export function Navbar({
  searchValue,
  onSearchChange,
  onSearchSubmit,
  brand = "BookDB",
  user,
}: NavbarProps) {
  const hasSearch = searchValue !== undefined || onSearchChange !== undefined;
  const showRightSide = hasSearch || !!user;
  const initials = user?.displayName
    .split(" ")
    .map((part) => part[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <header className="sticky top-0 z-40 bg-background border-b ">
      <div className="mx-auto flex max-w-5xl flex-col gap-3 px-4 py-2 lg:flex-row lg:items-center">
        <div className="flex items-center">
          <Link to="/">
            <img
              src="/logo.svg"
              alt={`${brand} logo`}
              className="h-auto w-28"
            />
          </Link>
        </div>
        {showRightSide && (
          <div className="ml-auto flex w-full items-center justify-end gap-2 sm:w-auto">
            {hasSearch && (
              <div className="w-full sm:w-90">
                <SearchBar
                  value={searchValue}
                  onChange={onSearchChange}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      onSearchSubmit?.(searchValue ?? "");
                    }
                  }}
                />
              </div>
            )}
            <div className="flex items-center gap-2">
              {user ? (
                <Link
                  to="/user/$username"
                  params={{ username: user.handle }}
                  aria-label="Profile"
                >
                  <Avatar size="lg">
                    {user.avatarUrl && (
                      <AvatarImage
                        src={user.avatarUrl}
                        alt={user.displayName}
                      />
                    )}
                    <AvatarFallback>{initials}</AvatarFallback>
                  </Avatar>
                </Link>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
