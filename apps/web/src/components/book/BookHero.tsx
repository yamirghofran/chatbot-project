import { useRef, useState, useEffect } from "react";
import type { Book, BookStats } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { AggregateStats } from "./AggregateStats";

export type BookHeroProps = {
  book: Book;
  stats?: BookStats;
};

export function BookHero({ book, stats }: BookHeroProps) {
  const [expanded, setExpanded] = useState(false);
  const [tagsExpanded, setTagsExpanded] = useState(false);
  const [clamped, setClamped] = useState(false);
  const descRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const el = descRef.current;
    if (el) {
      setClamped(el.scrollHeight > el.clientHeight);
    }
  }, [book.description]);

  return (
    <div>
      <div className="flex items-start justify-between gap-4">
        <h1 className="font-heading text-2xl font-semibold text-foreground">
          {book.title}
        </h1>
        {stats && <AggregateStats stats={stats} />}
      </div>
      <p className="text-muted-foreground mt-1">{book.author}</p>
      {book.description && (
        <div className="mt-3">
          <p
            ref={descRef}
            className={cn(
              "text-sm text-foreground leading-relaxed",
              !expanded && "line-clamp-3",
            )}
          >
            {book.description}
          </p>
          {(clamped || expanded) && (
            <button
              type="button"
              className="text-sm text-muted-foreground hover:text-foreground mt-1"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </div>
      )}
      {book.tags && book.tags.length > 0 && (
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {(tagsExpanded ? book.tags : book.tags.slice(0, 2)).map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
          {!tagsExpanded && book.tags.length > 2 && (
            <Badge asChild variant="secondary">
              <button
                type="button"
                className="cursor-pointer"
                onClick={() => setTagsExpanded(true)}
                aria-label={`Show ${book.tags.length - 2} more genres`}
              >
                +{book.tags.length - 2}
              </button>
            </Badge>
          )}
        </div>
      )}

    </div>
  );
}
