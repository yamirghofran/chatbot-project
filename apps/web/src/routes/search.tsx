import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { SearchPage } from "@/components/search/SearchPage";
import * as api from "@/lib/api";

type SearchParams = {
  q?: string;
};

export const Route = createFileRoute("/search")({
  validateSearch: (search: Record<string, unknown>): SearchParams => ({
    q: typeof search.q === "string" ? search.q : "",
  }),
  component: SearchRoute,
});

function SearchRoute() {
  const { q = "" } = Route.useSearch();
  const query = q.trim();

  const searchQuery = useQuery({
    queryKey: ["bookSearch", query],
    queryFn: () => api.searchBooks(query, 15),
    enabled: query.length > 0,
  });

  if (!query) {
    return (
      <p className="text-sm text-muted-foreground">
        Type a title or author in search.
      </p>
    );
  }

  if (searchQuery.isPending) {
    return (
      <div className="grid min-h-[50vh] place-items-center text-sm text-muted-foreground">
        Searching...
      </div>
    );
  }

  if (searchQuery.isError) {
    return (
      <p className="text-sm text-destructive">Could not load search results.</p>
    );
  }

  const books = searchQuery.data ?? [];
  const directHit = books[0];
  const keywordResults = books.slice(1);

  return (
    <SearchPage
      query={query}
      directHit={directHit}
      keywordResults={keywordResults}
    />
  );
}
