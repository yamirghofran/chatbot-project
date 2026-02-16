import type { Book, BookStats, ActivityItem, Review, User } from "@/lib/types";
import type { RatingPickerProps } from "./RatingPicker";
import { Separator } from "@/components/ui/separator";
import { BookHero } from "./BookHero";
import { BookRow } from "./BookRow";
import { FriendActivity } from "./FriendActivity";
import { ReviewList } from "./ReviewList";

export type BookPageProps = {
  book: Book;
  relatedBooks?: Book[];
  rating?: RatingPickerProps["value"];
  onRatingChange?: RatingPickerProps["onChange"];
  isLoved?: boolean;
  onLoveToggle?: () => void;
  onAddToList?: () => void;
  stats?: BookStats;
  friendActivity?: ActivityItem[];
  reviews?: Review[];
  currentUser?: User;
  onPostReview?: (text: string) => void;
  onLikeReview?: (reviewId: string) => void;
  onLikeReply?: (reviewId: string, replyId: string) => void;
  onReply?: (reviewId: string, text: string) => void;
};

export function BookPage({
  book,
  relatedBooks = [],
  rating,
  onRatingChange,
  isLoved,
  onLoveToggle,
  onAddToList,
  stats,
  friendActivity,
  reviews,
  currentUser,
  onPostReview,
  onLikeReview,
  onLikeReply,
  onReply,
}: BookPageProps) {
  return (
    <div>
      <BookHero
        book={book}
        rating={rating}
        onRatingChange={onRatingChange}
        isLoved={isLoved}
        onLoveToggle={onLoveToggle}
        onAddToList={onAddToList}
        stats={stats}
      />

      {(friendActivity?.length || relatedBooks.length > 0) && (
        <>
          <Separator className="my-6" />
          {friendActivity?.length && relatedBooks.length > 0 ? (
            <div className="grid grid-cols-2 gap-8">
              <FriendActivity items={friendActivity} />
              <div>
                <h2 className="font-heading text-lg font-semibold mb-2">Related</h2>
                <div>
                  {relatedBooks.map((b, i) => (
                    <div key={b.id}>
                      {i > 0 && <Separator />}
                      <BookRow book={b} variant="compact" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : friendActivity?.length ? (
            <FriendActivity items={friendActivity} />
          ) : (
            <>
              <h2 className="font-heading text-lg font-semibold mb-2">Related</h2>
              <div>
                {relatedBooks.map((b, i) => (
                  <div key={b.id}>
                    {i > 0 && <Separator />}
                    <BookRow book={b} variant="compact" />
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {reviews && (
        <>
          <Separator className="my-6" />
          <ReviewList
            reviews={reviews}
            currentUser={currentUser}
            onPostReview={onPostReview}
            onLikeReview={onLikeReview}
            onLikeReply={onLikeReply}
            onReply={onReply}
          />
        </>
      )}
    </div>
  );
}
