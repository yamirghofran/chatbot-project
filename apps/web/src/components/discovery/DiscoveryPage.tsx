import { useEffect, useRef, useState } from "react";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import type { Book, List, ActivityItem, User } from "@/lib/types";
import { BookRow } from "@/components/book/BookRow";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { StaffPicks } from "./StaffPicks";
import { ActivityFeed } from "./ActivityFeed";
import { TrendingLists } from "./TrendingLists";
import * as api from "@/lib/api";

export type DiscoveryPageProps = {
  books: Book[];
  currentUser?: User;
  userLists?: List[];
  staffPicks?: Book[];
  activity?: ActivityItem[];
  trendingLists?: List[];
};

export function DiscoveryPage({
  books,
  currentUser,
  userLists = [],
  staffPicks = [],
  activity = [],
  trendingLists = [],
}: DiscoveryPageProps) {
  const queryClient = useQueryClient();
  const [lists, setLists] = useState<List[]>(userLists);
  const [isProfileLinkCopied, setIsProfileLinkCopied] = useState(false);
  const nextListIdRef = useRef(userLists.length + 1);
  const copyResetTimerRef = useRef<number | null>(null);

  // Sync server state into local lists when react-query delivers it
  useEffect(() => {
    setLists(userLists);
  }, [userLists]);

  const toggleBookMutation = useMutation({
    mutationFn: ({ listId, bookId, nextSelected }: { listId: string; bookId: string; nextSelected: boolean }) =>
      nextSelected ? api.addBookToList(listId, bookId) : api.removeBookFromList(listId, bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myLists"] }),
  });

  const createListMutation = useMutation({
    mutationFn: async ({ name, book }: { name: string; book: Book }) => {
      const result = await api.createList(name);
      await api.addBookToList(result.id, book.id);
      return result;
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myLists"] }),
  });

  function selectedListIdsForBook(bookId: string) {
    return lists
      .filter((list) => list.books.some((listBook) => listBook.id === bookId))
      .map((list) => list.id);
  }

  function handleToggleBookInList(
    book: Book,
    listId: string,
    nextSelected: boolean,
  ) {
    // Optimistic update
    setLists((prevLists) =>
      prevLists.map((list) => {
        if (list.id !== listId) return list;
        const hasBook = list.books.some((b) => b.id === book.id);
        if (nextSelected && !hasBook) {
          return { ...list, books: [...list.books, book] };
        }
        if (!nextSelected && hasBook) {
          return { ...list, books: list.books.filter((b) => b.id !== book.id) };
        }
        return list;
      }),
    );
    toggleBookMutation.mutate({ listId, bookId: book.id, nextSelected });
  }

  function handleCreateListForBook(book: Book, name: string) {
    const trimmedName = name.trim();
    if (!trimmedName) return;

    // Optimistic update with temp ID
    const tempId = `l-local-${nextListIdRef.current++}`;
    setLists((prevLists) => [
      { id: tempId, name: trimmedName, owner: { id: "me", handle: "me", displayName: "You" }, books: [book] },
      ...prevLists,
    ]);
    createListMutation.mutate({ name: trimmedName, book });
  }

  async function handleShareProfile() {
    const profileUrl = `${window.location.origin}/user/${currentUser?.handle}`;
    try {
      await navigator.clipboard.writeText(profileUrl);
      setIsProfileLinkCopied(true);
      if (copyResetTimerRef.current) {
        window.clearTimeout(copyResetTimerRef.current);
      }
      copyResetTimerRef.current = window.setTimeout(() => {
        setIsProfileLinkCopied(false);
        copyResetTimerRef.current = null;
      }, 2000);
    } catch {
      setIsProfileLinkCopied(false);
    }
  }

  useEffect(() => {
    return () => {
      if (copyResetTimerRef.current) {
        window.clearTimeout(copyResetTimerRef.current);
      }
    };
  }, []);

  return (
    <div>
      <div className="flex gap-8 items-start">
        <section className="flex-1 min-w-0">
          <div className="mb-6 flex flex-col gap-5 rounded-xl bg-card p-5 sm:flex-row sm:items-center">
            <img
              src="/brand/cartoon-dancing.jpg"
              alt="Person reading with books"
              className="h-40 w-auto max-w-full rounded-lg object-contain"
            />
            <div className="min-w-0">
              <h2 className="font-heading text-3xl font-semibold text-foreground">
                Welcome back, {currentUser?.displayName?.split(" ")[0] ?? "there"}. What are you reading?
              </h2>
              <p className="mt-2 text-lg text-muted-foreground">
                Track a book to keep your library updated and get better picks.
              </p>
            </div>
          </div>
          <h2 className="font-heading text-lg font-semibold mb-2">
            Reccomended For You
          </h2>
          <div>
            {books.map((book, i) => (
              <div key={book.id}>
                {i > 0 && <Separator />}
                <BookRow
                  book={book}
                  showActions
                  tagVariant="discovery"
                  descriptionMode="preview"
                  primaryAction="amazon"
                  listOptions={lists}
                  selectedListIds={selectedListIdsForBook(book.id)}
                  onToggleList={(listId, nextSelected) =>
                    handleToggleBookInList(book, listId, nextSelected)
                  }
                  onCreateList={(name) => handleCreateListForBook(book, name)}
                />
              </div>
            ))}
          </div>
        </section>

        <aside className="w-72 shrink-0 space-y-6">
          {staffPicks.length > 0 && (
            <section>
              <h2 className="font-heading text-lg font-semibold mb-2">
                BookDB Picks
              </h2>
              <StaffPicks books={staffPicks} />
            </section>
          )}

          <section>
            <h2 className="font-heading text-lg font-semibold mb-2">
              Friend Activity
            </h2>
            {activity.length > 0 ? (
              <ActivityFeed items={activity} />
            ) : (
              <div className="rounded-lg bg-card px-3 py-2">
                <p className="text-sm text-muted-foreground">
                  It&apos;s quiet in here. Be the first to stir the shelves or
                  share your profile.
                </p>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-2"
                  onClick={handleShareProfile}
                >
                  {isProfileLinkCopied ? "Link copied!" : "Share your profile"}
                </Button>
              </div>
            )}
          </section>

          {trendingLists.length > 0 && (
            <section>
              <h2 className="font-heading text-lg font-semibold mb-2">
                Trending Lists
              </h2>
              <TrendingLists lists={trendingLists} />
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
