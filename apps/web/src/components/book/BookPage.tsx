import { ListPlus } from "lucide-react";
import { AmazonIcon } from "@/components/icons/AmazonIcon";
import type {
  Book,
  BookStats,
  ActivityItem,
  Review,
  User,
  List,
} from "@/lib/types";
import {
  RatingPicker,
  type RatingPickerProps,
} from "@/components/icons/RatingPicker";
import { TurtleShellIcon } from "@/components/icons/TurtleShellIcon";
import { AddToListMenu } from "@/components/list/AddToListMenu";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { BookCard } from "./BookCard";
import { BookHero } from "./BookHero";
import { FriendActivity } from "./FriendActivity";
import { ReviewList } from "./ReviewList";

export type BookPageProps = {
  book: Book;
  relatedBooks?: Book[];
  rating?: RatingPickerProps["value"];
  onRatingChange?: RatingPickerProps["onChange"];
  isShelled?: boolean;
  onShellToggle?: () => void;
  onAddToList?: () => void;
  listOptions?: List[];
  selectedListIds?: string[];
  onToggleList?: (listId: string, nextSelected: boolean) => void;
  onCreateList?: (name: string) => void;
  stats?: BookStats;
  friendActivity?: ActivityItem[];
  reviews?: Review[];
  totalReviews?: number;
  hasMoreReviews?: boolean;
  isLoadingMoreReviews?: boolean;
  onLoadMoreReviews?: () => void;
  currentUser?: User;
  onPostReview?: (text: string) => void;
  onLikeReview?: (reviewId: string) => void;
  onLikeReply?: (reviewId: string, replyId: string) => void;
  onReply?: (reviewId: string, text: string) => void;
  onDeleteReview?: (reviewId: string) => void;
  onDeleteReply?: (reviewId: string, replyId: string) => void;
};

export function BookPage({
  book,
  relatedBooks = [],
  rating,
  onRatingChange,
  isShelled,
  onShellToggle,
  onAddToList,
  listOptions,
  selectedListIds,
  onToggleList,
  onCreateList,
  stats,
  friendActivity,
  reviews,
  totalReviews,
  hasMoreReviews,
  isLoadingMoreReviews,
  onLoadMoreReviews,
  currentUser,
  onPostReview,
  onLikeReview,
  onLikeReply,
  onReply,
  onDeleteReview,
  onDeleteReply,
}: BookPageProps) {
  const amazonHref = `https://www.amazon.com/s?k=${encodeURIComponent(`${book.title} ${book.author} book`)}`;

  return (
    <div className="grid grid-cols-[1.1fr_4fr] gap-8">
      <div className="sticky top-28 lg:top-24 self-start">
        <img
          src={book.coverUrl ?? "/brand/book-placeholder.png"}
          alt={`Cover of ${book.title}`}
          className="w-full aspect-[2/3] rounded-sm object-cover"
        />

        <div className="flex flex-col items-center gap-2 mt-3 ">
          <div className="flex items-center gap-2 mb-1">
            <RatingPicker
              value={rating}
              onChange={onRatingChange}
              size="large"
            />
          </div>

          <div className="flex w-full gap-2 px-2">
            <Button
              type="button"
              variant={isShelled ? "secondary" : "default"}
              size="sm"
              className="flex-1"
              onClick={onShellToggle}
            >
              <TurtleShellIcon className="size-6" filled={isShelled} />
              {isShelled ? "In shell" : "Add to shell"}
            </Button>
            {listOptions ? (
              <AddToListMenu
                lists={listOptions.map((list) => ({
                  id: list.id,
                  name: list.name,
                  bookCount: list.books.length,
                }))}
                selectedListIds={selectedListIds}
                onToggleList={onToggleList}
                onCreateList={onCreateList}
                trigger={
                  <Button type="button" variant="outline" size="sm">
                    <ListPlus className="size-4" />
                  </Button>
                }
              />
            ) : (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={onAddToList}
              >
                <ListPlus className="size-4" />
              </Button>
            )}
          </div>
        </div>
        <Separator className=" mt-3 mb-2" />

        <a
          href={amazonHref}
          target="_blank"
          rel="noreferrer noopener"
          className="w-full  "
        >
          <Button type="button" variant="link" size="sm" className="w-full ">
            <AmazonIcon className="size-4" />
            Buy on Amazon
          </Button>
        </a>
      </div>

      <div>
        <BookHero book={book} stats={stats} />

        {(friendActivity?.length || relatedBooks.length > 0) && (
          <>
            <Separator className="my-6" />
            {friendActivity?.length && relatedBooks.length > 0 ? (
              <div className="grid grid-cols-2 gap-8">
                <FriendActivity items={friendActivity} />
                <div>
                  <h2 className="font-heading text-lg font-semibold mb-2">
                    Similar
                  </h2>
                  <div className="grid grid-cols-3 gap-3">
                    {relatedBooks.map((b) => (
                      <BookCard key={b.id} book={b} />
                    ))}
                  </div>
                </div>
              </div>
            ) : friendActivity?.length ? (
              <FriendActivity items={friendActivity} />
            ) : (
              <>
                <h2 className="font-heading text-lg font-semibold mb-2">
                  Similar
                </h2>
                <div className="grid grid-cols-6 gap-3">
                  {relatedBooks.map((b) => (
                    <BookCard key={b.id} book={b} />
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
              totalReviews={totalReviews}
              hasMore={hasMoreReviews}
              isLoadingMore={isLoadingMoreReviews}
              onLoadMore={onLoadMoreReviews}
              currentUser={currentUser}
              onPostReview={onPostReview}
              onLikeReview={onLikeReview}
              onLikeReply={onLikeReply}
              onReply={onReply}
              onDeleteReview={onDeleteReview}
              onDeleteReply={onDeleteReply}
            />
          </>
        )}
      </div>
    </div>
  );
}
