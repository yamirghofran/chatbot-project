import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BookPage } from "@/components/book/BookPage";
import { CenteredLoading } from "@/components/ui/CenteredLoading";
import * as api from "@/lib/api";
import { useMyLists } from "@/lib/useMyLists";
import { useCurrentUser } from "@/lib/auth";
import type { Book, Review } from "@/lib/types";

export const Route = createFileRoute("/books/$bookId")({
  component: BookDetailPage,
});

const REVIEWS_LIMIT = 20;

function BookDetailPage() {
  const { bookId } = Route.useParams();
  const queryClient = useQueryClient();
  const { data: me } = useCurrentUser();
  const myRatingQueryKey = ["myRating", bookId];

  // ---------------------------------------------------------------------------
  // Reviews — accumulated in local state so "load more" can append pages
  // without fighting the query cache.
  // ---------------------------------------------------------------------------
  const [reviews, setReviews] = useState<Review[]>([]);
  const [totalReviews, setTotalReviews] = useState(0);
  const [reviewsLoaded, setReviewsLoaded] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);

  const reviewsQuery = useQuery({
    queryKey: ["bookReviews", bookId],
    queryFn: () => api.getBookReviews(bookId, REVIEWS_LIMIT, 0),
  });

  // Sync page-0 query data → local state (also resets after any invalidation).
  useEffect(() => {
    if (reviewsQuery.data) {
      setReviews(reviewsQuery.data.items);
      setTotalReviews(reviewsQuery.data.total);
      setReviewsLoaded(reviewsQuery.data.items.length);
    }
  }, [reviewsQuery.data]);

  const hasMore = reviewsLoaded < totalReviews;

  async function handleLoadMore() {
    setLoadingMore(true);
    try {
      const data = await api.getBookReviews(bookId, REVIEWS_LIMIT, reviewsLoaded);
      setReviews((prev) => [...prev, ...data.items]);
      setTotalReviews(data.total);
      setReviewsLoaded((prev) => prev + data.items.length);
    } finally {
      setLoadingMore(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Other queries
  // ---------------------------------------------------------------------------

  const bookQuery = useQuery({
    queryKey: ["book", bookId],
    queryFn: () => api.getBook(bookId),
  });

  const relatedQuery = useQuery({
    queryKey: ["relatedBooks", bookId],
    queryFn: () => api.getRelatedBooks(bookId),
  });

  const myShellQuery = useQuery({
    queryKey: ["myShell"],
    queryFn: () => api.getMyShell(),
    enabled: !!me,
  });

  const myRatingQuery = useQuery({
    queryKey: myRatingQueryKey,
    queryFn: () => api.getMyRating(bookId),
    enabled: !!me,
  });

  const { lists, selectedListIds, toggleBook, createListForBook } = useMyLists(bookId);

  // ---------------------------------------------------------------------------
  // Rating / shell mutations (unchanged)
  // ---------------------------------------------------------------------------

  const ratingMutation = useMutation({
    mutationFn: (rating: number | undefined) =>
      rating === undefined
        ? api.deleteRating(bookId)
        : api.upsertRating(bookId, rating),
    onMutate: async (nextRating) => {
      await queryClient.cancelQueries({ queryKey: myRatingQueryKey });
      const previous = queryClient.getQueryData<{ rating: number | null }>(myRatingQueryKey);
      queryClient.setQueryData(myRatingQueryKey, { rating: nextRating ?? null });
      return { previous };
    },
    onError: (_error, _nextRating, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(myRatingQueryKey, context.previous);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ["book", bookId] });
      await queryClient.invalidateQueries({ queryKey: myRatingQueryKey });
      await queryClient.invalidateQueries({ queryKey: ["userFavorites", me?.handle] });
      await queryClient.invalidateQueries({ queryKey: ["userActivity", me?.handle] });
      await queryClient.invalidateQueries({ queryKey: ["activityFeed"] });
    },
  });

  const shellMutation = useMutation({
    mutationFn: (nextShelled: boolean) =>
      nextShelled ? api.addToShell(bookId) : api.removeFromShell(bookId),
    onMutate: async (nextShelled) => {
      await queryClient.cancelQueries({ queryKey: ["myShell"] });
      const previousShell = queryClient.getQueryData<Book[]>(["myShell"]);
      const nextShell = [...(previousShell ?? [])];
      if (nextShelled) {
        const alreadyShelled = nextShell.some((item) => item.id === bookId);
        if (!alreadyShelled && bookQuery.data) nextShell.unshift(bookQuery.data);
      } else {
        queryClient.setQueryData(["myShell"], nextShell.filter((item) => item.id !== bookId));
        return { previousShell };
      }
      queryClient.setQueryData(["myShell"], nextShell);
      return { previousShell };
    },
    onError: (_error, _nextShelled, context) => {
      if (context?.previousShell) queryClient.setQueryData(["myShell"], context.previousShell);
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ["myShell"] });
      await queryClient.invalidateQueries({ queryKey: ["book", bookId] });
      await queryClient.invalidateQueries({ queryKey: ["userActivity", me?.handle] });
      await queryClient.invalidateQueries({ queryKey: ["activityFeed"] });
    },
  });

  // ---------------------------------------------------------------------------
  // Review mutations — optimistic updates go directly to local `reviews` state.
  // On settle we invalidate page-0 so the useEffect resets with fresh server data.
  // ---------------------------------------------------------------------------

  const postReviewMutation = useMutation({
    mutationFn: (text: string) => api.postReview(bookId, text),
    onMutate: (text) => {
      if (!me) return;
      const optimistic: Review = {
        id: `optimistic-${Date.now()}`,
        user: me,
        text,
        likes: 0,
        isLikedByMe: false,
        timestamp: "just now",
        replies: [],
      };
      setReviews((prev) => [optimistic, ...prev]);
      setTotalReviews((t) => t + 1);
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
  });

  const deleteReviewMutation = useMutation({
    mutationFn: (reviewId: string) => api.deleteReview(reviewId),
    onMutate: (reviewId) => {
      setReviews((prev) => prev.filter((r) => r.id !== reviewId));
      setTotalReviews((t) => Math.max(0, t - 1));
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
  });

  const likeReviewMutation = useMutation({
    mutationFn: ({ reviewId, liked }: { reviewId: string; liked: boolean }) =>
      liked ? api.likeReview(reviewId) : api.unlikeReview(reviewId),
    onMutate: ({ reviewId, liked }) => {
      setReviews((prev) =>
        prev.map((r) =>
          r.id === reviewId
            ? { ...r, isLikedByMe: liked, likes: r.likes + (liked ? 1 : -1) }
            : r
        )
      );
    },
    onError: (_err, { reviewId, liked }) => {
      // Roll back the toggle
      setReviews((prev) =>
        prev.map((r) =>
          r.id === reviewId
            ? { ...r, isLikedByMe: !liked, likes: r.likes + (!liked ? 1 : -1) }
            : r
        )
      );
    },
  });

  const postCommentMutation = useMutation({
    mutationFn: ({ reviewId, text }: { reviewId: string; text: string }) =>
      api.postReviewComment(reviewId, text),
    onMutate: ({ reviewId, text }) => {
      if (!me) return;
      const optimisticReply = {
        id: `optimistic-reply-${Date.now()}`,
        user: me,
        text,
        likes: 0,
        isLikedByMe: false as const,
        timestamp: "just now",
      };
      setReviews((prev) =>
        prev.map((r) =>
          r.id === reviewId ? { ...r, replies: [...(r.replies ?? []), optimisticReply] } : r
        )
      );
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: ({ reviewId, commentId }: { reviewId: string; commentId: string }) =>
      api.deleteReviewComment(reviewId, commentId),
    onMutate: ({ reviewId, commentId }) => {
      setReviews((prev) =>
        prev.map((r) =>
          r.id === reviewId
            ? { ...r, replies: (r.replies ?? []).filter((reply) => reply.id !== commentId) }
            : r
        )
      );
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["bookReviews", bookId] });
    },
  });

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  if (bookQuery.isLoading) return <CenteredLoading />;
  if (bookQuery.isError || !bookQuery.data) {
    return <div className="py-8 text-destructive text-sm">Book not found.</div>;
  }

  const book = bookQuery.data;
  const stats = book.stats ?? undefined;
  const isShelled = (myShellQuery.data ?? []).some((b) => b.id === bookId);
  const rating = myRatingQuery.data?.rating ?? undefined;

  return (
    <BookPage
      book={book}
      stats={stats}
      rating={rating}
      isShelled={isShelled}
      relatedBooks={relatedQuery.data ?? []}
      reviews={reviews}
      totalReviews={totalReviews}
      hasMoreReviews={hasMore}
      isLoadingMoreReviews={loadingMore}
      currentUser={me}
      listOptions={lists}
      selectedListIds={selectedListIds}
      onToggleList={(listId, nextSelected) => toggleBook(listId, nextSelected)}
      onCreateList={(name) => createListForBook(name)}
      onRatingChange={(rating) => ratingMutation.mutate(rating)}
      onShellToggle={() => shellMutation.mutate(!isShelled)}
      onPostReview={(text) => postReviewMutation.mutate(text)}
      onDeleteReview={(reviewId) => deleteReviewMutation.mutate(reviewId)}
      onLikeReview={(reviewId) => {
        const review = reviews.find((r) => r.id === reviewId);
        likeReviewMutation.mutate({ reviewId, liked: !review?.isLikedByMe });
      }}
      onReply={(reviewId, text) => postCommentMutation.mutate({ reviewId, text })}
      onDeleteReply={(reviewId, replyId) =>
        deleteCommentMutation.mutate({ reviewId, commentId: replyId })
      }
      onLoadMoreReviews={handleLoadMore}
    />
  );
}
