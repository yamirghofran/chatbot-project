import { Link } from "@tanstack/react-router";
import type { Book } from "@/lib/types";

export type BookCardProps = {
  book: Book;
};

export function BookCard({ book }: BookCardProps) {
  return (
    <Link
      to="/books/$bookId"
      params={{ bookId: book.id }}
      className="group block"
    >
      <img
        src={book.coverUrl ?? "/brand/book-placeholder.png"}
        alt={`Cover of ${book.title}`}
        className="w-full aspect-[2/3] rounded-sm object-cover"
      />
      <p className="mt-2 text-sm font-medium text-foreground line-clamp-2 group-hover:underline leading-snug">
        {book.title}
      </p>
      <p className="mt-0.5 text-xs text-muted-foreground line-clamp-1">
        {book.author}
      </p>
    </Link>
  );
}
