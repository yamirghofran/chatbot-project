import type { Book } from "@/lib/types";
import { BookRow } from "@/components/book/BookRow";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";

const PREVIEW_COUNT = 3;

export type FavoritesListProps = {
  books: Book[];
  onViewAll?: () => void;
};

export function FavoritesList({ books, onViewAll }: FavoritesListProps) {
  if (books.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4">No favourites yet.</p>
    );
  }

  const preview = books.slice(0, PREVIEW_COUNT);
  const remaining = books.length - PREVIEW_COUNT;

  return (
    <div>
      {preview.map((book, i) => (
        <div key={book.id}>
          {i > 0 && <Separator />}
          <BookRow book={book} variant="compact" />
        </div>
      ))}
      {remaining > 0 && (
        <Button
          variant="ghost"
          size="sm"
          className="w-full mt-2 text-muted-foreground"
          onClick={onViewAll}
        >
          View all {books.length} favourites
        </Button>
      )}
    </div>
  );
}
