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
  isLoved?: boolean;
  onLoveToggle?: () => void;
  onAddToList?: () => void;
};

export function BookPage({
  book,
  relatedBooks = [],
  rating,
  onRatingChange,
  isLoved,
  onLoveToggle,
  onAddToList,
}: BookPageProps) {
  return (
    <div>
      <BookHero
        book={book}
        rating={rating}
        onRatingChange={onRatingChange}
        isLoved={isLoved}
        onLoveToggle={onLoveToggle}
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
