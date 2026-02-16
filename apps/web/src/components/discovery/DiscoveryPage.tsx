import type { Book, List, ActivityItem } from "@/lib/types";
import { BookRow } from "@/components/book/BookRow";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { BookPlus, ListPlus } from "lucide-react";
import { StaffPicks } from "./StaffPicks";
import { ActivityFeed } from "./ActivityFeed";
import { TrendingLists } from "./TrendingLists";

export type DiscoveryPageProps = {
  books: Book[];
  staffPicks?: Book[];
  activity?: ActivityItem[];
  trendingLists?: List[];
};

export function DiscoveryPage({
  books,
  staffPicks = [],
  activity = [],
  trendingLists = [],
}: DiscoveryPageProps) {
  return (
    <div>
      <div className="flex gap-8 items-start">
        <section className="flex-1 min-w-0">
          <div className="mb-6 flex flex-col gap-5 rounded-xl border bg-card p-5 sm:flex-row sm:items-center">
            <img
              src="/brand/cartoon-dancing.jpg"
              alt="Person reading with books"
              className="h-40 w-auto max-w-full rounded-lg object-contain"
            />
            <div className="min-w-0">
              <h2 className="font-heading text-2xl font-semibold text-foreground">
                Welcome back, Matt. What are you reading?
              </h2>
              <p className="mt-2 text-base text-muted-foreground">
                Track a book to keep your library updated and get better picks.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button size="sm" type="button">
                  <BookPlus className="mr-1" />
                  Track a book
                </Button>
                <Button size="sm" variant="outline" type="button">
                  <ListPlus className="mr-1" />
                  Start a list
                </Button>
              </div>
            </div>
          </div>
          <h2 className="font-heading text-lg font-semibold mb-2">
            Reccomended For You
          </h2>
          <div>
            {books.map((book, i) => (
              <div key={book.id}>
                {i > 0 && <Separator />}
                <BookRow book={book} showActions />
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
