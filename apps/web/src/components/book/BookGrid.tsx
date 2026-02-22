import type { Book } from "@/lib/types";
import { cn } from "@/lib/utils";
import { BookCard } from "./BookCard";

export type BookGridProps = {
  books: Book[];
  columns?: 3 | 4 | 5 | 6;
  className?: string;
};

const colClass: Record<NonNullable<BookGridProps["columns"]>, string> = {
  3: "grid-cols-3",
  4: "grid-cols-4",
  5: "grid-cols-5",
  6: "grid-cols-6",
};

export function BookGrid({ books, columns = 5, className }: BookGridProps) {
  if (books.length === 0) return null;

  return (
    <div className={cn("grid gap-x-4 gap-y-6", colClass[columns], className)}>
      {books.map((book) => (
        <BookCard key={book.id} book={book} />
      ))}
    </div>
  );
}
