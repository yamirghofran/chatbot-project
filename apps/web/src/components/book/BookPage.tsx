import { ListPlus } from "lucide-react";
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
import { BookHero } from "./BookHero";
import { BookRow } from "./BookRow";
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
  currentUser,
  onPostReview,
  onLikeReview,
  onLikeReply,
  onReply,
}: BookPageProps) {
  return (
    <div className="grid grid-cols-[1fr_4fr] gap-8">
      <div className="sticky top-6 self-start">
        <img
          src={book.coverUrl ?? "/brand/book-placeholder.png"}
          alt={`Cover of ${book.title}`}
          className="w-full aspect-[2/3] rounded-sm object-cover"
        />
        <div className="flex flex-col items-center gap-2 mt-3">
          <div className="flex items-center gap-1">
            <RatingPicker
              value={rating}
              onChange={onRatingChange}
              size="large"
            />
          </div>
          <Separator className="my-1 w-full" />
          <Button
            type="button"
            variant={isShelled ? "secondary" : "default"}
            size="sm"
            className="w-full"
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
              className="w-full"
              trigger={
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  <ListPlus className="size-4" />
                  Add to list
                </Button>
              }
            />
          ) : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="w-full"
              onClick={onAddToList}
            >
              <ListPlus className="size-4" />
              Add to list
            </Button>
          )}
        </div>
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
                <h2 className="font-heading text-lg font-semibold mb-2">
                  Similar
                </h2>
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
    </div>
  );
}
