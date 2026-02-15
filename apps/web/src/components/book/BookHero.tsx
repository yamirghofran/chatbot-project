import { Heart, ListPlus } from "lucide-react";
import type { Book } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { RatingPicker, type RatingPickerProps } from "./RatingPicker";

export type BookHeroProps = {
  book: Book;
  rating?: RatingPickerProps["value"];
  onRatingChange?: RatingPickerProps["onChange"];
  isFavorited?: boolean;
  onFavoriteToggle?: () => void;
  onAddToList?: () => void;
};

export function BookHero({
  book,
  rating,
  onRatingChange,
  isFavorited = false,
  onFavoriteToggle,
  onAddToList,
}: BookHeroProps) {
  return (
    <div className="flex gap-6 items-start">
      <img
        src={book.coverUrl}
        alt={`Cover of ${book.title}`}
        className="w-44 aspect-[2/3] rounded-sm object-cover shrink-0"
      />
      <div className="min-w-0">
        <h1 className="font-heading text-2xl font-semibold text-foreground">
          {book.title}
        </h1>
        <p className="text-muted-foreground mt-1">{book.author}</p>
        {book.description && (
          <p className="text-sm text-foreground mt-3 leading-relaxed">
            {book.description}
          </p>
        )}
        {book.tags && book.tags.length > 0 && (
          <div className="flex gap-1.5 mt-3 flex-wrap">
            {book.tags.slice(0, 2).map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
            {book.tags.length > 2 && (
              <Badge variant="secondary">+{book.tags.length - 2}</Badge>
            )}
          </div>
        )}

        <div className="flex items-center gap-4 mt-4">
          <RatingPicker value={rating} onChange={onRatingChange} />
          <button
            type="button"
            className="size-9 inline-flex items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground"
            onClick={onFavoriteToggle}
            aria-label={isFavorited ? "Remove from favourites" : "Add to favourites"}
          >
            <Heart
              className={cn("size-5", isFavorited && "fill-red-500 text-red-500")}
            />
          </button>
          <button
            type="button"
            className="size-9 inline-flex items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground"
            onClick={onAddToList}
            aria-label="Add to list"
          >
            <ListPlus className="size-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
