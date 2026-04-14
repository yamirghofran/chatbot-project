import { Link } from "@tanstack/react-router";
import type { Book } from "@/lib/types";

const SOURCE_LABELS: Record<string, string> = {
  vector_search: "Search",
  database: "Database",
  vector_similarity: "Similar",
  cold_start: "Popular",
  mcp: "MCP Engine",
  comparison: "Comparison",
  "bpr+vector": "For You",
  bpr: "For You",
  vector: "For You",
};

function getSourceLabel(source?: string): string | undefined {
  if (!source) return undefined;
  if (SOURCE_LABELS[source]) return SOURCE_LABELS[source];
  if (source.startsWith("local_fallback")) return "Recommended";
  return undefined;
}

export type InlineBookCardProps = {
  book: Book;
  source?: string;
};

export function InlineBookCard({ book, source }: InlineBookCardProps) {
  const label = getSourceLabel(source);

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
        {label && (
          <span className="inline-block mt-1 text-[10px] leading-none px-1.5 py-0.5 rounded-full bg-accent/50 text-muted-foreground">
            {label}
          </span>
        )}
      </div>
    </Link>
  );
}
