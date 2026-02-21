import { Sparkles, ArrowUp, Star, MessageCircle } from "lucide-react";
import { Link } from "@tanstack/react-router";
import type { Book } from "@/lib/types";
import { BookGrid } from "@/components/book/BookGrid";
import { SearchResultList } from "./SearchResultRow";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";

export type SearchPageProps = {
  query: string;
  directHit?: Book;
  keywordResults?: Book[];
  aiNarrative?: string;
  aiBooks?: Book[];
  isAiLoading?: boolean;
  followUpValue?: string;
  onFollowUpChange?: (value: string) => void;
  onFollowUpSubmit?: (value: string) => void;
  followUpSuggestions?: string[];
};

// ─── Shared formatting ───────────────────────────────────────────────────────

const compactNumber = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

// ─── Direct hit card ─────────────────────────────────────────────────────────

function DirectHitCard({ book }: { book: Book }) {
  const averageRating = book.averageRating?.toFixed(1);
  const ratingCount = book.ratingCount ?? 0;
  const commentCount = book.commentCount ?? 0;
  const shellCount = book.shellCount ?? 0;
  const tags = book.tags?.slice(0, 3) ?? [];

  return (
    <Link
      to="/books/$bookId"
      params={{ bookId: book.id }}
      className="group flex items-center gap-4  rounded-lg py-4 px-1 transition-colors"
    >
      <img
        src={book.coverUrl ?? "/brand/book-placeholder.png"}
        alt={`Cover of ${book.title}`}
        className="h-20 w-auto aspect-[2/3] rounded-sm object-cover shrink-0"
      />

      <div className="flex-1 min-w-0">
        <p className="text-base font-semibold text-foreground leading-snug">
          {book.title}
        </p>
        <p className="text-sm text-muted-foreground mt-0.5">{book.author}</p>
        {tags.length > 0 && (
          <p className="text-xs text-muted-foreground/60 mt-1">
            {tags.join(" · ")}
          </p>
        )}
      </div>

      <div className="flex flex-col items-end gap-1.5 text-xs text-muted-foreground shrink-0 tabular-nums">
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

// ─── AI skeleton ─────────────────────────────────────────────────────────────

function AiSkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="space-y-2">
        <div className="h-3.5 bg-input rounded w-3/4" />
        <div className="h-3.5 bg-input rounded w-full" />
        <div className="h-3.5 bg-input rounded w-5/6" />
      </div>
      <div className="grid grid-cols-5 gap-x-4 gap-y-6 mt-5">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i}>
            <div className="w-full aspect-[2/3] bg-input rounded-sm" />
            <div className="mt-2 h-3 bg-input rounded w-4/5" />
            <div className="mt-1 h-2.5 bg-input rounded w-3/5" />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function SearchPage({
  query,
  directHit,
  keywordResults = [],
  aiNarrative,
  aiBooks = [],
  isAiLoading = false,
  followUpValue = "",
  onFollowUpChange,
  onFollowUpSubmit,
  followUpSuggestions,
}: SearchPageProps) {
  const hasAiContent = isAiLoading || !!aiNarrative || aiBooks.length > 0;
  const hasMoreResults = keywordResults.length > 0;
  const hasAnything = directHit || hasAiContent || hasMoreResults;

  function handleFollowUpKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && followUpValue.trim()) {
      onFollowUpSubmit?.(followUpValue.trim());
    }
  }

  return (
    <div className="space-y-6">
      {/* 0. Header */}
      <p className="text-sm text-muted-foreground">
        Showing results for{" "}
        <span className="font-medium text-foreground">
          &ldquo;{query}&rdquo;
        </span>
      </p>

      {/* 1. Direct hit */}
      {directHit && <DirectHitCard book={directHit} />}

      {/* 2. AI section */}
      {hasAiContent && (
        <section className="rounded-xl bg-input/20 px-6 py-6 space-y-6">
          {/* Query echo — right side */}
          <p className="text-right text-sm text-muted-foreground/70 italic">
            &ldquo;{query}&rdquo;
          </p>

          {/* AI response — left side */}
          {isAiLoading && !aiNarrative ? (
            <AiSkeleton />
          ) : (
            <div className="flex gap-3">
              <Sparkles className="size-4 text-primary shrink-0 mt-0.5" />
              <div className="space-y-5 min-w-0">
                {aiNarrative && (
                  <p className="text-sm text-foreground leading-relaxed">
                    {aiNarrative}
                  </p>
                )}
                {aiBooks.length > 0 && <BookGrid books={aiBooks} />}
              </div>
            </div>
          )}

          {/* Follow-up input */}
          <div className="space-y-3 pt-2  border-border/50">
            {followUpSuggestions && followUpSuggestions.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {followUpSuggestions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => {
                      onFollowUpChange?.(s);
                      onFollowUpSubmit?.(s);
                    }}
                    className="text-xs px-3 py-1.5 rounded-full border border-input bg-background text-muted-foreground hover:text-foreground hover:border-ring transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <div className="relative">
              <input
                type="text"
                value={followUpValue}
                onChange={(e) => onFollowUpChange?.(e.target.value)}
                onKeyDown={handleFollowUpKeyDown}
                placeholder="Refine — longer? Darker? Different era?"
                className="w-full rounded-[12px] supports-[corner-shape:squircle]:rounded-[70px] supports-[corner-shape:squircle]:[corner-shape:squircle] border border-input bg-background py-2.5 pl-4 pr-10 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
              <button
                type="button"
                onClick={() => {
                  if (followUpValue.trim())
                    onFollowUpSubmit?.(followUpValue.trim());
                }}
                aria-label="Submit follow-up"
                className="absolute right-2 top-1/2 -translate-y-1/2 size-7 inline-flex items-center justify-center rounded-full bg-primary text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-40"
                disabled={!followUpValue.trim()}
              >
                <ArrowUp className="size-3.5" />
              </button>
            </div>
          </div>
        </section>
      )}

      {/* 4. More results */}
      {hasMoreResults && (
        <section className="space-y-2">
          <p className="text-xs text-muted-foreground uppercase tracking-wide"></p>
          <SearchResultList books={keywordResults} />
        </section>
      )}

      {/* Empty state */}
      {!hasAnything && (
        <p className="text-sm text-muted-foreground">
          No results for &ldquo;{query}&rdquo;.
        </p>
      )}
    </div>
  );
}
