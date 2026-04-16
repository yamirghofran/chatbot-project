import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { BookGrid } from "@/components/book/BookGrid";
import { CenteredLoading } from "@/components/ui/CenteredLoading";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";
import * as api from "@/lib/api";

export const Route = createFileRoute("/user/$username/shell")({
  component: UserShellPage,
});

function UserShellPage() {
  const { username } = Route.useParams();

  const userQuery = useQuery({
    queryKey: ["user", username],
    queryFn: () => api.getUser(username),
  });

  const shellQuery = useQuery({
    queryKey: ["userShell", username],
    queryFn: () => api.getUserShell(username),
  });

  if (shellQuery.isLoading || userQuery.isLoading) {
    return <CenteredLoading />;
  }

  const user = userQuery.data;
  const books = shellQuery.data ?? [];

  return (
    <div className="w-full">
      <Link
        to="/user/$username"
        params={{ username }}
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="size-4" />
        Back to profile
      </Link>

      <div className="flex items-center gap-3 mb-6">
        <TurtleShellIcon className="size-7 text-foreground" />
        <h1 className="font-heading text-2xl font-bold">
          {user?.displayName ?? username}&apos;s Shell
        </h1>
      </div>

      {books.length > 0 ? (
        <BookGrid books={books} columns={5} />
      ) : (
        <p className="text-sm text-muted-foreground py-8">
          No books in shell yet.
        </p>
      )}
    </div>
  );
}
