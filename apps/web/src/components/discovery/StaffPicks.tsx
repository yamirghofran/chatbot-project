import type { Book } from "@/lib/types";

export type StaffPicksProps = {
  books: Book[];
};

export function StaffPicks({ books }: StaffPicksProps) {
  return (
    <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
      {books.map((book) => (
        <img
          key={book.id}
          src={book.coverUrl}
          alt={`Cover of ${book.title}`}
          className="w-24 aspect-[2/3] rounded-sm object-cover shrink-0"
        />
      ))}
    </div>
  );
}
