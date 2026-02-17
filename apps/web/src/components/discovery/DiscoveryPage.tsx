import { useRef, useState } from "react";
import type { Book, List, ActivityItem } from "@/lib/types";
import { BookRow } from "@/components/book/BookRow";
import { Separator } from "@/components/ui/separator";
import { StaffPicks } from "./StaffPicks";
import { ActivityFeed } from "./ActivityFeed";
import { TrendingLists } from "./TrendingLists";

export type DiscoveryPageProps = {
  books: Book[];
  userLists?: List[];
  staffPicks?: Book[];
  activity?: ActivityItem[];
  trendingLists?: List[];
};

export function DiscoveryPage({
  books,
  userLists = [],
  staffPicks = [],
  activity = [],
  trendingLists = [],
}: DiscoveryPageProps) {
  const [lists, setLists] = useState<List[]>(userLists);
  const nextListIdRef = useRef(userLists.length + 1);

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
  }

  function handleCreateListForBook(book: Book, name: string) {
    const trimmedName = name.trim();
    if (!trimmedName) return;

    const newList: List = {
      id: `l-local-${nextListIdRef.current++}`,
      name: trimmedName,
      owner: {
        id: "me",
        handle: "me",
        displayName: "You",
      },
      books: [book],
    };

    setLists((prevLists) => [newList, ...prevLists]);
  }

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
                Welcome back, Matt. What are you reading?
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

          {activity.length > 0 && (
            <section>
              <h2 className="font-heading text-lg font-semibold mb-2">
                Friend Activity
              </h2>
              <ActivityFeed items={activity} />
            </section>
          )}

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
