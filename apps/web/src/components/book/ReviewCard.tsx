import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { Trash2 } from "lucide-react";
import { ThumbsUpIcon } from "@/components/icons/ThumbsUpIcon";
import type { Review, Reply } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";

export type ReviewCardProps = {
  review: Review | Reply;
  isReply?: boolean;
  onLike?: () => void;
  onReply?: () => void;
  onDelete?: () => void;
  isOwnReview?: boolean;
};

const MAX_REVIEW_LENGTH = 420;

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function ReviewCard({ review, isReply = false, onLike, onReply, onDelete, isOwnReview }: ReviewCardProps) {
  const [expanded, setExpanded] = useState(false);
  const isLongReview = review.text.length > MAX_REVIEW_LENGTH;
  const visibleText =
    isLongReview && !expanded
      ? `${review.text.slice(0, MAX_REVIEW_LENGTH).trimEnd()}...`
      : review.text;

  return (
    <div className={cn("flex gap-3", isReply && "ml-10")}>
      <Avatar size={isReply ? "sm" : "default"}>
        <AvatarFallback>{getInitials(review.user.displayName)}</AvatarFallback>
      </Avatar>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <Link
            to="/user/$username"
            params={{ username: review.user.handle }}
            className="text-sm font-medium text-foreground hover:underline"
          >
            {review.user.displayName}
          </Link>
          <span className="text-xs text-muted-foreground">{review.timestamp}</span>
          {isOwnReview && (
            <button
              type="button"
              onClick={onDelete}
              className="ml-auto text-muted-foreground hover:text-destructive transition-colors"
              aria-label="Delete"
            >
              <Trash2 className="size-3.5" />
            </button>
          )}
        </div>
        <div className="relative mt-1">
          <p className="text-sm text-foreground leading-relaxed">{visibleText}</p>
          {isLongReview && !expanded && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-background to-transparent"
            />
          )}
        </div>
        {isLongReview && !expanded && (
          <button
            type="button"
            className="mt-1 text-xs font-medium text-muted-foreground hover:text-foreground hover:underline"
            onClick={() => setExpanded(true)}
          >
            Continue reading
          </button>
        )}
        <div className="flex items-center gap-2 mt-1.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={onLike}
            className={cn(
              review.isLikedByMe ? "text-foreground" : "text-muted-foreground"
            )}
          >
            <ThumbsUpIcon className="size-3" />
            {review.likes > 0 && review.likes}
          </Button>
          {!isReply && (
            <Button
              variant="ghost"
              size="xs"
              className="text-muted-foreground"
              onClick={onReply}
            >
              Reply
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
