import { useLayoutEffect, useRef } from "react";
import { ListPlus, MessageCircle, Star } from "lucide-react";
import type { Book, List } from "@/lib/types";
import { cn } from "@/lib/utils";
import { AmazonIcon } from "@/components/icons/AmazonIcon";
import { ShellButton } from "@/components/icons/ShellButton";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";
import { Badge } from "@/components/ui/badge";
import { AddToListMenu } from "@/components/list/AddToListMenu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export type BookRowProps = {
  book: Book;
  variant?: "default" | "compact";
  showActions?: boolean;
  descriptionMode?: "none" | "preview";
  showDescriptionPreview?: boolean;
  primaryAction?: "shell" | "amazon";
  onShellToggle?: () => void;
  onAddToList?: () => void;
  listOptions?: List[];
  selectedListIds?: string[];
  onToggleList?: (listId: string, nextSelected: boolean) => void;
  onCreateList?: (name: string) => void;
  isShelled?: boolean;
};

const actionBtnClass =
  "size-8 inline-flex cursor-pointer items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground";

const compactNumber = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function deriveEngagement(book: Book) {
  const base = Number.parseInt(book.id, 10) || book.title.length;
  const averageRating = 3.6 + (base % 14) / 10;
  const ratingCount = 140 + base * 37;
  const commentCount = 42 + base * 11;
  const shellCount = 24 + base * 5;

  return {
    averageRating,
    ratingCount,
    commentCount,
    shellCount,
  };
}

export function BookRow({
  book,
  variant = "default",
  showActions = false,
  descriptionMode = "none",
  showDescriptionPreview = false,
  primaryAction = "shell",
  onShellToggle,
  onAddToList,
  listOptions,
  selectedListIds,
  onToggleList,
  onCreateList,
  isShelled = false,
}: BookRowProps) {
  const isCompact = variant === "compact";
  const amazonHref = `https://www.amazon.com/s?k=${encodeURIComponent(`${book.title} ${book.author} book`)}`;
  const engagement = deriveEngagement(book);
  const shouldShowDescriptionPreview =
    !isCompact &&
    !!book.description &&
    (descriptionMode === "preview" || showDescriptionPreview);
  const contentRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useLayoutEffect(() => {
    if (isCompact) return;
    if (!contentRef.current || !imageRef.current) return;

    const contentEl = contentRef.current;
    const imageEl = imageRef.current;

    const syncImageHeight = () => {
      const nextHeight = contentEl.getBoundingClientRect().height;
      imageEl.style.height = `${Math.max(1, Math.round(nextHeight))}px`;
    };

    syncImageHeight();

    const observer = new ResizeObserver(() => {
      syncImageHeight();
    });
    observer.observe(contentEl);

    return () => {
      observer.disconnect();
      imageEl.style.height = "";
    };
  }, [isCompact, showDescriptionPreview, book.description, book.tags, book.author, book.title]);

  return (
    <div
      className={cn(
        "flex cursor-pointer gap-4 py-3",
        isCompact ? "items-center gap-3 py-2" : "items-start",
      )}
    >
      <img
        ref={imageRef}
        src={book.coverUrl}
        alt={`Cover of ${book.title}`}
        className={cn(
          "rounded-[12px] supports-[corner-shape:squircle]:rounded-[15px] supports-[corner-shape:squircle]:[corner-shape:squircle] object-cover shrink-0",
          isCompact
            ? "h-10 w-7"
            : "aspect-[2/3] w-auto",
        )}
      />

      <div ref={contentRef} className="flex-1 min-w-0">
        <p
          className={cn(
            "font-medium text-foreground truncate",
            isCompact ? "text-sm" : "text-base",
          )}
        >
          {book.title}
        </p>
        <p
          className={cn(
            "text-muted-foreground truncate",
            isCompact ? "text-xs" : "text-sm",
          )}
        >
          {book.author}
        </p>
        {!isCompact && book.tags && book.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {book.tags.slice(0, 2).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[11px]">
                {tag}
              </Badge>
            ))}
            {book.tags.length > 2 && (
              <Badge variant="secondary" className="text-[11px]">
                +{book.tags.length - 2}
              </Badge>
            )}
          </div>
        )}
        {shouldShowDescriptionPreview && (
          <p className="mt-1.5 w-[90%] line-clamp-3 text-xs leading-relaxed text-muted-foreground">
            {book.description}
          </p>
        )}
        {!isCompact && (
          <div className="mt-1.5 flex items-center gap-3 text-[11px] text-muted-foreground/85">
            <span className="inline-flex items-center gap-1">
              <Star className="size-3" />
              {engagement.averageRating.toFixed(1)} (
              {compactNumber.format(engagement.ratingCount)})
            </span>
            <span className="inline-flex items-center gap-1">
              <MessageCircle className="size-3" />
              {compactNumber.format(engagement.commentCount)}
            </span>
            <span className="inline-flex items-center gap-1">
              <TurtleShellIcon className="size-4" />
              {compactNumber.format(engagement.shellCount)}
            </span>
          </div>
        )}
      </div>

      {showActions && (
        <TooltipProvider>
          <div className="flex self-center items-center gap-2 shrink-0">
            {primaryAction === "shell" ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <ShellButton
                    isShelled={isShelled}
                    onClick={onShellToggle}
                    className={actionBtnClass}
                  />
                </TooltipTrigger>
                <TooltipContent>Add to shell</TooltipContent>
              </Tooltip>
            ) : (
              <a
                href={amazonHref}
                target="_blank"
                rel="noreferrer noopener"
                className={actionBtnClass}
                aria-label={`Find ${book.title} on Amazon`}
              >
                <AmazonIcon className="size-4" />
              </a>
            )}
            {listOptions ? (
              <AddToListMenu
                align="end"
                lists={listOptions.map((list) => ({
                  id: list.id,
                  name: list.name,
                  bookCount: list.books.length,
                }))}
                selectedListIds={selectedListIds}
                onToggleList={onToggleList}
                onCreateList={onCreateList}
                trigger={
                  <button
                    type="button"
                    className={actionBtnClass}
                    aria-label="Add to list"
                  >
                    <ListPlus className="size-4" />
                  </button>
                }
              />
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className={actionBtnClass}
                    onClick={onAddToList}
                    aria-label="Add to list"
                  >
                    <ListPlus className="size-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Add to list</TooltipContent>
              </Tooltip>
            )}
          </div>
        </TooltipProvider>
      )}
    </div>
  );
}
