import { Link } from "@tanstack/react-router";
import type { List } from "@/lib/types";

export type ListRowProps = {
  list: List;
};

export function ListRow({ list }: ListRowProps) {
  const previewBooks = list.books.slice(0, 5);

  return (
    <div className="flex items-center gap-4 py-3">
      <div className="flex -space-x-2 shrink-0">
        {previewBooks.map((book) => (
          <img
            key={book.id}
            src={book.coverUrl ?? "/brand/book-placeholder.png"}
            alt={`Cover of ${book.title}`}
            className="h-10 w-7 rounded-sm object-cover border-2 border-background"
          />
        ))}
      </div>
      <div className="min-w-0">
        <Link
          to="/lists/$listId"
          params={{ listId: list.id }}
          className="text-sm font-medium text-foreground truncate hover:underline"
        >
          {list.name}
        </Link>
        <p className="text-xs text-muted-foreground truncate">
          {list.books.length} books &middot; {list.owner.displayName}
        </p>
      </div>
    </div>
  );
}
