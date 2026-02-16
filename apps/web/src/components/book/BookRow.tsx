import { Heart, ListPlus } from "lucide-react";
import type { Book } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
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
  onLoveToggle?: () => void;
  onAddToList?: () => void;
  isLoved?: boolean;
};

const actionBtnClass =
  "size-8 inline-flex items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground";

export function BookRow({
  book,
  variant = "default",
  showActions = false,
  onLoveToggle,
  onAddToList,
  isLoved = false,
}: BookRowProps) {
  const isCompact = variant === "compact";

  return (
    <div
      className={cn("flex items-center gap-4 py-3", isCompact && "gap-3 py-2")}
    >
      <img
        src={book.coverUrl}
        alt={`Cover of ${book.title}`}
        className={cn(
          "rounded-sm object-cover shrink-0",
          isCompact ? "h-10 w-7" : "h-20 w-16",
        )}
      />

      <div className="flex-1 min-w-0">
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
          <div className="flex gap-1 mt-1.5">
            {book.tags.slice(0, 2).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
            {book.tags.length > 2 && (
              <Badge variant="secondary" className="text-xs">
                +{book.tags.length - 2}
              </Badge>
            )}
          </div>
        )}
      </div>

      {showActions && (
        <TooltipProvider>
          <div className="flex items-center gap-2 shrink-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className={actionBtnClass}
                  onClick={onLoveToggle}
                  aria-label={isLoved ? "Remove from loved" : "Love"}
                >
                  <Heart
                    className={cn(
                      "size-4",
                      isLoved && "fill-red-500 text-red-500",
                    )}
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>Love</TooltipContent>
            </Tooltip>
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
          </div>
        </TooltipProvider>
      )}
    </div>
  );
}
