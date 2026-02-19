import { useState, useMemo } from "react";
import { Star, ArrowUp, ArrowDown, X } from "lucide-react";
import type { RatedBook } from "@/lib/types";
import { Toggle } from "@/components/ui/toggle";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { RatedBookRow } from "./RatedBookList";

export type AllBooksPageProps = {
  ratedBooks: RatedBook[];
};

const STAR_OPTIONS = [5, 4, 3, 2, 1] as const;

type SortKey = "rating" | "date" | "title";
type SortDir = "asc" | "desc";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "date", label: "Date rated" },
  { key: "rating", label: "Rating" },
  { key: "title", label: "Title" },
];

function sortBooks(
  books: RatedBook[],
  key: SortKey,
  dir: SortDir,
): RatedBook[] {
  const sorted = [...books];
  sorted.sort((a, b) => {
    switch (key) {
      case "rating":
        return a.rating - b.rating;
      case "date":
        return a.ratedAt.localeCompare(b.ratedAt);
      case "title":
        return a.book.title.localeCompare(b.book.title);
    }
  });
  if (dir === "desc") sorted.reverse();
  return sorted;
}

export function AllBooksPage({ ratedBooks }: AllBooksPageProps) {
  const [activeFilters, setActiveFilters] = useState<Set<number>>(new Set());
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  function toggleFilter(star: number) {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(star)) {
        next.delete(star);
      } else {
        next.add(star);
      }
      return next;
    });
  }

  function cycleSort() {
    const currentIndex = SORT_OPTIONS.findIndex((o) => o.key === sortKey);
    const next = SORT_OPTIONS[(currentIndex + 1) % SORT_OPTIONS.length];
    setSortKey(next.key);
  }

  const results = useMemo(() => {
    const filtered =
      activeFilters.size === 0
        ? ratedBooks
        : ratedBooks.filter((rb) => activeFilters.has(Math.round(rb.rating)));
    return sortBooks(filtered, sortKey, sortDir);
  }, [ratedBooks, activeFilters, sortKey, sortDir]);

  function getCount(star: number) {
    return ratedBooks.filter((rb) => Math.round(rb.rating) === star).length;
  }

  const activeSortLabel = SORT_OPTIONS.find((o) => o.key === sortKey)!.label;

  return (
    <div>
      <h1 className="font-heading text-xl font-semibold text-foreground mb-4">
        Library ({results.length})
      </h1>

      <div className="flex flex-wrap items-center gap-2 mb-2">
        {STAR_OPTIONS.map((star) => {
          const count = getCount(star);
          if (count === 0) return null;
          return (
            <Toggle
              key={star}
              variant="outline"
              size="sm"
              pressed={activeFilters.has(star)}
              onPressedChange={() => toggleFilter(star)}
            >
              <Star className="size-3.5 text-[#FFCC00] fill-[#FFCC00]" />
              {star} ({count})
            </Toggle>
          );
        })}
        {activeFilters.size > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
            onClick={() => setActiveFilters(new Set())}
          >
            <X className="size-3.5" />
            Clear
          </Button>
        )}
      </div>

      <div className="flex items-center gap-1 mb-4">
        <span className="text-sm text-muted-foreground">Sort by</span>
        <Button variant="ghost" size="sm" onClick={cycleSort}>
          {activeSortLabel}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
          aria-label={sortDir === "asc" ? "Sort descending" : "Sort ascending"}
        >
          {sortDir === "desc" ? (
            <ArrowDown className="size-3.5" />
          ) : (
            <ArrowUp className="size-3.5" />
          )}
        </Button>
      </div>

      {results.length === 0 ? (
        <div className="flex flex-col items-center rounded-xl  bg-card p-8">
          <img src="/brand/cartoon-sitting.jpg" alt="" className="w-32" />
          <p className="text-sm text-muted-foreground mt-4">
            No books match the selected filters.
          </p>
        </div>
      ) : (
        <div>
          {results.map((rb, i) => (
            <div key={rb.book.id}>
              {i > 0 && <Separator />}
              <RatedBookRow ratedBook={rb} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
