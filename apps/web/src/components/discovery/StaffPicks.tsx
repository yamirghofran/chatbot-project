import { Link } from "@tanstack/react-router";
import type { Book } from "@/lib/types";

export type StaffPicksProps = {
  books: Book[];
};

export function StaffPicks({ books }: StaffPicksProps) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {books.slice(0, 3).map((book, i) => (
        <div key={book.id}>
          <Link to="/books/$bookId" params={{ bookId: book.id }} className="group relative block cursor-pointer overflow-hidden rounded-sm">
            <img
              src={book.coverUrl ?? "/brand/book-placeholder.png"}
              alt={`Cover of ${book.title}`}
              className="w-full aspect-[2/3] object-cover transition-[filter] duration-(--duration-normal) ease-(--ease-in-out) group-hover:brightness-[0.35]"
            />
            <div className="absolute inset-0 flex flex-col justify-end p-2 opacity-0 transition-opacity duration-(--duration-normal) ease-(--ease-in-out) group-hover:opacity-100">
              <p className="text-[11px] font-semibold leading-tight text-white">{book.title}</p>
              <p className="mt-0.5 text-[10px] text-white/70">{book.author}</p>
            </div>
          </Link>
          <p className="mt-1 text-xs text-muted-foreground">#{i + 1}</p>
        </div>
      ))}
    </div>
  );
}
