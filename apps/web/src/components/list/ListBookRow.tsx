import type { Book } from "@/lib/types";
import { cn } from "@/lib/utils";

export type ListBookRowProps = {
  book: Book;
  compact?: boolean;
};

export function ListBookRow({ book, compact = false }: ListBookRowProps) {
  return (
    <div
      className={cn("flex items-center gap-3 py-2", compact && "gap-2 py-1.5")}
    >
      <img
        src={book.coverUrl ?? "/brand/book-placeholder.png"}
        alt={`Cover of ${book.title}`}
        className={cn(
          "rounded-[10px] supports-[corner-shape:squircle]:rounded-[15px] supports-[corner-shape:squircle]:[corner-shape:squircle] object-cover shrink-0",
          compact ? "h-12 w-8.5" : "h-16 w-11",
        )}
      />
      <div className="min-w-0">
        <p
          className={cn(
            "truncate font-medium text-foreground",
            compact ? "text-sm" : "text-base",
          )}
        >
          {book.title}
        </p>
        <p
          className={cn(
            "truncate text-muted-foreground",
            compact ? "text-xs" : "text-sm",
          )}
        >
          {book.author}
        </p>
      </div>
    </div>
  );
}
