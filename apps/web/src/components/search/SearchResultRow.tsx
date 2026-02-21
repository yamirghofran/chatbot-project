import { Link } from "@tanstack/react-router";
import { Star, MessageCircle } from "lucide-react";
import type { Book } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";

export type SearchResultRowProps = {
  book: Book;
};

const compactNumber = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function deriveEngagement(book: Book) {
  const seed = Number.parseInt(book.id, 10) || book.title.length;
  return {
    averageRating: (3.6 + (seed % 14) / 10).toFixed(1),
    ratingCount: 140 + seed * 37,
    commentCount: 42 + seed * 11,
    shellCount: 24 + seed * 5,
  };
}

export function SearchResultRow({ book }: SearchResultRowProps) {
  const { averageRating, ratingCount, commentCount, shellCount } =
    deriveEngagement(book);
  const tags = book.tags?.slice(0, 2) ?? [];

  return (
    <Link
      to="/books/$bookId"
      params={{ bookId: book.id }}
      className="group flex items-center gap-4 py-3 hover:bg-muted/40 -mx-2 px-2 rounded-md transition-colors"
    >
      <img
        src={book.coverUrl ?? "/brand/book-placeholder.png"}
        alt={`Cover of ${book.title}`}
        className="h-14 w-auto aspect-[2/3] rounded-sm object-cover shrink-0"
      />

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground truncate">
          {book.title}
        </p>
        <p className="text-xs text-muted-foreground truncate mt-0.5">
          {book.author}
          {tags.length > 0 && (
            <span className="text-muted-foreground/60">
              {" · "}
              {tags.join(" · ")}
            </span>
          )}
        </p>
      </div>

      <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0 tabular-nums">
        <span className="flex items-center gap-1">
          <Star className="size-3 fill-[#FFCC00] text-[#FFCC00]" />
          {averageRating}
          <span className="text-muted-foreground/60">
            ({compactNumber.format(ratingCount)})
          </span>
        </span>
        <span className="flex items-center gap-1">
          <MessageCircle className="size-3" />
          {compactNumber.format(commentCount)}
        </span>
        <span className="flex items-center gap-1">
          <TurtleShellIcon className="size-3.5" />
          {compactNumber.format(shellCount)}
        </span>
      </div>
    </Link>
  );
}

export type SearchResultListProps = {
  books: Book[];
};

export function SearchResultList({ books }: SearchResultListProps) {
  if (books.length === 0) return null;
  return (
    <div>
      {books.map((book, i) => (
        <div key={book.id}>
          {i > 0 && <Separator />}
          <SearchResultRow book={book} />
        </div>
      ))}
    </div>
  );
}
