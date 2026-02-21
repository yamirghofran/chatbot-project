import { Link } from "@tanstack/react-router";
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
};

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function ReviewCard({ review, isReply = false, onLike, onReply }: ReviewCardProps) {
  return (
    <div className={cn("flex gap-3", isReply && "ml-10")}>
      <Avatar size={isReply ? "sm" : "default"}>
        <AvatarFallback>{getInitials(review.user.displayName)}</AvatarFallback>
      </Avatar>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <Link
            to="/users/$userId"
            params={{ userId: review.user.id }}
            className="text-sm font-medium text-foreground hover:underline"
          >
            {review.user.displayName}
          </Link>
          <span className="text-xs text-muted-foreground">{review.timestamp}</span>
        </div>
        <p className="text-sm text-foreground mt-1 leading-relaxed">{review.text}</p>
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
