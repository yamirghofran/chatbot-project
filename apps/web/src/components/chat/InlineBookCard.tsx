import { Link } from "@tanstack/react-router";
import type { Book } from "@/lib/types";

export function InlineBookCard({ book }: { book: Book }) {
  return (
    <Link
      to="/books/$bookId"
      params={{ bookId: book.id }}
      className="flex items-center gap-3 rounded-lg border border-border/60 bg-background px-3 py-2 hover:border-ring/40 transition-colors min-w-[220px] max-w-[280px] shrink-0"
    >
      <img
        src={book.coverUrl ?? "/brand/book-placeholder.png"}
        alt={`Cover of ${book.title}`}
        className="h-14 w-auto aspect-[2/3] rounded-sm object-cover shrink-0"
      />
      <div className="min-w-0">
        <p className="text-sm font-medium text-foreground line-clamp-2 leading-snug">
          {book.title}
        </p>
        <p className="text-xs text-muted-foreground line-clamp-1 mt-0.5">
          {book.author}
        </p>
      </div>
    </Link>
  );
}
