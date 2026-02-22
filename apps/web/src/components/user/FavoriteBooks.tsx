import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Search, X } from "lucide-react";
import type { Book } from "@/lib/types";
import * as api from "@/lib/api";

export type FavoriteBooksProps = {
  books: Book[];
  username: string;
  isOwnProfile?: boolean;
  isEditing?: boolean;
};

export function FavoriteBooks({
  books,
  username,
  isOwnProfile = false,
  isEditing = false,
}: FavoriteBooksProps) {
  const queryClient = useQueryClient();
  const [activeSlot, setActiveSlot] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    const id = window.setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 250);
    return () => window.clearTimeout(id);
  }, [query]);

  const searchQuery = useQuery({
    queryKey: ["favoriteSearch", debouncedQuery],
    queryFn: () => api.searchBooks(debouncedQuery, 8),
    enabled: isOwnProfile && isEditing && debouncedQuery.length >= 2,
  });

  const updateFavoritesMutation = useMutation({
    mutationFn: async ({
      replaceBookId,
      nextBookId,
    }: {
      replaceBookId?: string;
      nextBookId: string;
    }) => {
      if (replaceBookId && replaceBookId !== nextBookId) {
        await api.removeFromMyFavorites(replaceBookId);
      }
      await api.addToMyFavorites(nextBookId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["userFavorites", username],
      });
      await queryClient.invalidateQueries({ queryKey: ["myFavorites"] });
      setQuery("");
      setActiveSlot(null);
    },
  });

  const removeFavoriteMutation = useMutation({
    mutationFn: (bookId: string) => api.removeFromMyFavorites(bookId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["userFavorites", username],
      });
      await queryClient.invalidateQueries({ queryKey: ["myFavorites"] });
    },
  });

  const slots = useMemo<(Book | undefined)[]>(
    () => [books[0], books[1], books[2]],
    [books],
  );

  const visibleResults = useMemo(() => {
    const currentSlotBookId =
      activeSlot !== null ? slots[activeSlot]?.id : undefined;
    const data = searchQuery.data;
    const mergedResults = [
      data?.directHit ?? null,
      ...(data?.keywordResults ?? []),
      ...(data?.aiBooks ?? []),
    ].filter((result): result is Book => Boolean(result));
    const dedupedResults = Array.from(
      new Map(mergedResults.map((result) => [result.id, result])).values(),
    );
    return dedupedResults.filter((result) => {
      if (result.id === currentSlotBookId) return true;
      return !books.some((book) => book.id === result.id);
    });
  }, [searchQuery.data, books, activeSlot, slots]);

  const isSearching =
    debouncedQuery.length >= 2 &&
    (searchQuery.isPending || searchQuery.isFetching);

  if (!isEditing) {
    if (books.length === 0) {
      return (
        <p className="text-sm text-muted-foreground py-4">
          No favorites picked yet.
        </p>
      );
    }

    return (
      <div className="flex gap-4">
        {books.slice(0, 3).map((book) => (
          <div key={book.id} className="w-28 shrink-0">
            <img
              src={book.coverUrl ?? "/brand/book-placeholder.png"}
              alt={`Cover of ${book.title}`}
              className="w-28 h-[168px] rounded-md object-cover"
            />
            <p className="mt-1.5 text-sm font-medium text-foreground truncate">
              {book.title}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              {book.author}
            </p>
          </div>
        ))}
      </div>
    );
  }

  function handleSelectBook(book: Book) {
    if (activeSlot === null) return;
    const current = slots[activeSlot];
    updateFavoritesMutation.mutate({
      replaceBookId: current?.id,
      nextBookId: book.id,
    });
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-4">
        {slots.map((book, index) => (
          <div key={book?.id ?? `slot-${index}`} className="w-28 shrink-0">
            <button
              type="button"
              className="relative block w-28"
              disabled={!isOwnProfile || !isEditing}
              onClick={() => {
                if (!isOwnProfile || !isEditing) return;
                setActiveSlot(index);
              }}
            >
              {book ? (
                <img
                  src={book.coverUrl ?? "/brand/book-placeholder.png"}
                  alt={`Cover of ${book.title}`}
                  className="w-28 h-[168px] rounded-md object-cover"
                />
              ) : (
                <div className="flex h-[168px] w-28 items-center justify-center rounded-md border border-dashed border-border bg-muted/50 text-muted-foreground">
                  <Plus className="size-5" />
                </div>
              )}
              {isOwnProfile && isEditing && book && (
                <span
                  role="button"
                  tabIndex={0}
                  className="absolute right-1 top-1 inline-flex size-5 items-center justify-center rounded-full bg-background/90 text-muted-foreground shadow-sm"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    removeFavoriteMutation.mutate(book.id);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      removeFavoriteMutation.mutate(book.id);
                    }
                  }}
                >
                  <X className="size-3" />
                </span>
              )}
            </button>
            {book ? (
              <>
                <p className="mt-1.5 text-sm font-medium text-foreground truncate">
                  {book.title}
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  {book.author}
                </p>
              </>
            ) : (
              <p className="mt-1.5 text-xs text-muted-foreground"></p>
            )}
          </div>
        ))}
      </div>

      {isOwnProfile && isEditing && activeSlot !== null && (
        <div className="rounded-md border border-border bg-background p-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2 top-2.5 size-4 text-muted-foreground" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search books..."
              className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-2 text-sm outline-none ring-offset-background focus-visible:ring-1 focus-visible:ring-ring"
            />
          </div>
          <div className="mt-2 max-h-56 overflow-auto">
            {debouncedQuery.length < 2 ? (
              <p className="py-2 text-xs text-muted-foreground">
                Type at least 2 characters.
              </p>
            ) : isSearching ? (
              <p className="py-2 text-xs text-muted-foreground">
                Loading books...
              </p>
            ) : searchQuery.isError ? (
              <p className="py-2 text-xs text-destructive">
                Could not load results.
              </p>
            ) : visibleResults.length === 0 ? (
              <p className="py-2 text-xs text-muted-foreground">
                No matching books.
              </p>
            ) : (
              visibleResults.map((result) => (
                <button
                  key={result.id}
                  type="button"
                  className="flex w-full items-center gap-2 rounded-md px-1.5 py-1.5 text-left hover:bg-muted/60"
                  onClick={() => handleSelectBook(result)}
                >
                  <img
                    src={result.coverUrl ?? "/brand/book-placeholder.png"}
                    alt={`Cover of ${result.title}`}
                    className="h-10 w-7 rounded-sm object-cover"
                  />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-foreground">
                      {result.title}
                    </p>
                    <p className="truncate text-xs text-muted-foreground">
                      {result.author}
                    </p>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
