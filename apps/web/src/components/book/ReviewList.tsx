import { useState } from "react";
import type { Review, User } from "@/lib/types";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ReviewCard } from "./ReviewCard";

export type ReviewListProps = {
  reviews: Review[];
  currentUser?: User;
  onPostReview?: (text: string) => void;
  onLikeReview?: (reviewId: string) => void;
  onLikeReply?: (reviewId: string, replyId: string) => void;
  onReply?: (reviewId: string, text: string) => void;
};

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function ReviewList({
  reviews,
  currentUser,
  onPostReview,
  onLikeReview,
  onLikeReply,
  onReply,
}: ReviewListProps) {
  const [reviewText, setReviewText] = useState("");
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyText, setReplyText] = useState("");

  function handlePostReview() {
    if (reviewText.trim() && onPostReview) {
      onPostReview(reviewText.trim());
      setReviewText("");
    }
  }

  function handlePostReply(reviewId: string) {
    if (replyText.trim() && onReply) {
      onReply(reviewId, replyText.trim());
      setReplyText("");
      setReplyingTo(null);
    }
  }

  return (
    <div>
      <h2 className="font-heading text-lg font-semibold mb-4">
        Reviews{reviews.length > 0 && ` (${reviews.length})`}
      </h2>

      {currentUser && (
        <div className="flex gap-3 mb-6">
          <Avatar>
            <AvatarFallback>{getInitials(currentUser.displayName)}</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <textarea
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-none"
              placeholder="Write a review..."
              rows={3}
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
            />
            <div className="flex justify-end mt-2">
              <Button size="sm" onClick={handlePostReview} disabled={!reviewText.trim()}>
                Post
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-0">
        {reviews.map((review, i) => (
          <div key={review.id}>
            {i > 0 && <Separator className="my-4" />}
            <div className="space-y-3">
              <ReviewCard
                review={review}
                onLike={() => onLikeReview?.(review.id)}
                onReply={() =>
                  setReplyingTo(replyingTo === review.id ? null : review.id)
                }
              />

              {review.replies?.map((reply) => (
                <ReviewCard
                  key={reply.id}
                  review={reply}
                  isReply
                  onLike={() => onLikeReply?.(review.id, reply.id)}
                />
              ))}

              {replyingTo === review.id && (
                <div className="ml-10 flex gap-3">
                  {currentUser && (
                    <Avatar size="sm">
                      <AvatarFallback>
                        {getInitials(currentUser.displayName)}
                      </AvatarFallback>
                    </Avatar>
                  )}
                  <div className="flex-1 min-w-0">
                    <textarea
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-none"
                      placeholder="Write a reply..."
                      rows={2}
                      value={replyText}
                      onChange={(e) => setReplyText(e.target.value)}
                      autoFocus
                    />
                    <div className="flex justify-end gap-2 mt-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setReplyingTo(null);
                          setReplyText("");
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handlePostReply(review.id)}
                        disabled={!replyText.trim()}
                      >
                        Reply
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
