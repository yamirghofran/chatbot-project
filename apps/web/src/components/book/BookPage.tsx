import type { Book } from "@/lib/types";
import type { RatingPickerProps } from "./RatingPicker";
import { Separator } from "@/components/ui/separator";
import { BookHero } from "./BookHero";
import { BookRow } from "./BookRow";

export type BookPageProps = {
  book: Book;
  relatedBooks?: Book[];
  rating?: RatingPickerProps["value"];
  onRatingChange?: RatingPickerProps["onChange"];
  isFavorited?: boolean;
  onFavoriteToggle?: () => void;
  onAddToList?: () => void;
};

export function BookPage({
  book,
  relatedBooks = [],
  rating,
  onRatingChange,
  isFavorited,
  onFavoriteToggle,
  onAddToList,
}: BookPageProps) {
  return (
    <div>
      <BookHero
        book={book}
        rating={rating}
        onRatingChange={onRatingChange}
        isFavorited={isFavorited}
        onFavoriteToggle={onFavoriteToggle}
        onAddToList={onAddToList}
      />

      {relatedBooks.length > 0 && (
        <>
          <Separator className="my-6" />
          <h2 className="font-heading text-lg font-semibold mb-2">Related</h2>
          <div>
            {relatedBooks.map((b, i) => (
              <div key={b.id}>
                {i > 0 && <Separator />}
                <BookRow book={b} variant="compact" />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
