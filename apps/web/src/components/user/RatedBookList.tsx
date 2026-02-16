import { Star } from "lucide-react";
import type { RatedBook } from "@/lib/types";
import { BookRow } from "@/components/book/BookRow";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";

const PREVIEW_COUNT = 5;

export type RatedBookListProps = {
  ratedBooks: RatedBook[];
  onViewAll?: () => void;
};

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1 shrink-0">
      <Star className="size-3.5 text-[#FFCC00] fill-[#FFCC00]" />
      <span className="text-sm font-medium text-foreground">{rating}</span>
    </div>
  );
}

export function RatedBookRow({ ratedBook }: { ratedBook: RatedBook }) {
  return (
    <div className="flex items-center">
      <div className="flex-1 min-w-0">
        <BookRow book={ratedBook.book} />
      </div>
      <StarRating rating={ratedBook.rating} />
    </div>
  );
}

export function RatedBookList({ ratedBooks, onViewAll }: RatedBookListProps) {
  if (ratedBooks.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4">No rated books yet.</p>
    );
  }

  const preview = ratedBooks.slice(0, PREVIEW_COUNT);
  const remaining = ratedBooks.length - PREVIEW_COUNT;

  return (
    <div>
      {preview.map((rb, i) => (
        <div key={rb.book.id}>
          {i > 0 && <Separator />}
          <RatedBookRow ratedBook={rb} />
        </div>
      ))}
      {remaining > 0 && (
        <Button
          variant="ghost"
          size="sm"
          className="w-full mt-2 text-muted-foreground"
          onClick={onViewAll}
        >
          View full library ({ratedBooks.length})
        </Button>
      )}
    </div>
  );
}
