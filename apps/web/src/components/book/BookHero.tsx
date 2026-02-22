import { useRef, useState, useEffect } from "react";
import type { Book, BookStats } from "@/lib/types";
import { cn } from "@/lib/utils";
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
      <h1 className="font-heading text-2xl font-semibold text-foreground">
        {book.title}
      </h1>
      <p className="text-muted-foreground mt-1 mb-3">
        {book.author}
        {book.publicationYear && (
          <span className="text-muted-foreground/60 before:content-['·'] before:mx-1.5">
            {book.publicationYear}
          </span>
        )}
      </p>
      {stats && <AggregateStats stats={stats} />}
      {book.description && (
        <div className="mt-3">
          <p
            ref={descRef}
            className={cn(
              "text-sm text-foreground leading-relaxed",
              !expanded && "line-clamp-6",
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
        <p className="mt-3 text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Genres: </span>
          {(tagsExpanded ? book.tags : book.tags.slice(0, 5)).join(" · ")}
          {!tagsExpanded && book.tags.length > 5 && (
            <>
              {" · "}
              <button
                type="button"
                className="hover:text-foreground"
                onClick={() => setTagsExpanded(true)}
              >
                +{book.tags.length - 5}
              </button>
            </>
          )}
        </p>
      )}
    </div>
  );
}
