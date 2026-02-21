import type { Book } from "@/lib/types";

export type FavoriteBooksProps = {
  books: Book[];
};

export function FavoriteBooks({ books }: FavoriteBooksProps) {
  if (books.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4">No favorites picked yet.</p>
    );
  }

  return (
    <div className="flex gap-4">
      {books.slice(0, 3).map((book) => (
        <div key={book.id} className="w-28 shrink-0">
          <img
            src={book.coverUrl ?? "/brand/book-placeholder.png"}
            alt={`Cover of ${book.title}`}
            className="w-28 h-[168px] rounded-md object-cover"
          />
          <p className="mt-1.5 text-sm font-medium text-foreground truncate">{book.title}</p>
          <p className="text-xs text-muted-foreground truncate">{book.author}</p>
        </div>
      ))}
    </div>
  );
}
