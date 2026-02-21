import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BookPage } from "@/components/book/BookPage";
import * as api from "@/lib/api";
import { useMyLists } from "@/lib/useMyLists";
import { useCurrentUser } from "@/lib/auth";
import type { Book, RatedBook } from "@/lib/types";

export const Route = createFileRoute("/books/$bookId")({
  component: BookDetailPage,
});

function BookDetailPage() {
  const { bookId } = Route.useParams();
  const queryClient = useQueryClient();
  const { data: me } = useCurrentUser();
  const ratingsQueryKey = ["userRatings", me?.handle];

  const bookQuery = useQuery({
    queryKey: ["book", bookId],
    queryFn: () => api.getBook(bookId),
  });

  const reviewsQuery = useQuery({
    queryKey: ["bookReviews", bookId],
    queryFn: () => api.getBookReviews(bookId),
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

  const myRatingsQuery = useQuery({
    queryKey: ["userRatings", me?.handle],
    queryFn: () => api.getUserRatings(me!.handle, 200, "recent"),
    enabled: !!me?.handle,
  });

  const { lists, selectedListIds, toggleBook, createListForBook } = useMyLists(bookId);

  const ratingMutation = useMutation({
    mutationFn: (rating: number | undefined) =>
      rating === undefined
        ? api.deleteRating(bookId)
        : api.upsertRating(bookId, rating),
    onMutate: async (nextRating) => {
      await queryClient.cancelQueries({ queryKey: ratingsQueryKey });
      const previousRatings = queryClient.getQueryData<RatedBook[]>(ratingsQueryKey);

      const nextRatings = [...(previousRatings ?? [])];
      const existingIndex = nextRatings.findIndex((entry) => entry.book.id === bookId);
      if (nextRating === undefined) {
        if (existingIndex >= 0) {
          nextRatings.splice(existingIndex, 1);
        }
      } else if (existingIndex >= 0) {
        nextRatings[existingIndex] = { ...nextRatings[existingIndex], rating: nextRating };
      } else if (bookQuery.data) {
        nextRatings.unshift({
          book: bookQuery.data,
          rating: nextRating,
          ratedAt: new Date().toISOString(),
        });
      }
      queryClient.setQueryData(ratingsQueryKey, nextRatings);
      return { previousRatings };
    },
    onError: (_error, _nextRating, context) => {
      if (context?.previousRatings) {
        queryClient.setQueryData(ratingsQueryKey, context.previousRatings);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ["book", bookId] });
      await queryClient.invalidateQueries({ queryKey: ratingsQueryKey });
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
        if (!alreadyShelled && bookQuery.data) {
          nextShell.unshift(bookQuery.data);
        }
      } else {
        const filtered = nextShell.filter((item) => item.id !== bookId);
        queryClient.setQueryData(["myShell"], filtered);
        return { previousShell };
      }

      queryClient.setQueryData(["myShell"], nextShell);
      return { previousShell };
    },
    onError: (_error, _nextShelled, context) => {
      if (context?.previousShell) {
        queryClient.setQueryData(["myShell"], context.previousShell);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ["myShell"] });
      await queryClient.invalidateQueries({ queryKey: ["book", bookId] });
      await queryClient.invalidateQueries({ queryKey: ["userActivity", me?.handle] });
      await queryClient.invalidateQueries({ queryKey: ["activityFeed"] });
    },
  });

  if (bookQuery.isLoading) {
    return <div className="py-8 text-muted-foreground text-sm">Loading…</div>;
  }

  if (bookQuery.isError || !bookQuery.data) {
    return <div className="py-8 text-destructive text-sm">Book not found.</div>;
  }

  const book = bookQuery.data;
  const stats = book.stats ?? undefined;
  const isShelled = (myShellQuery.data ?? []).some((b) => b.id === bookId);
  const rating = myRatingsQuery.data?.find((entry) => entry.book.id === bookId)?.rating;

  return (
    <BookPage
      book={book}
      stats={stats}
      rating={rating}
      isShelled={isShelled}
      relatedBooks={relatedQuery.data ?? []}
      reviews={reviewsQuery.data ?? []}
      listOptions={lists}
      selectedListIds={selectedListIds}
      onToggleList={(listId, nextSelected) => toggleBook(listId, nextSelected)}
      onCreateList={(name) => createListForBook(name)}
      onRatingChange={(rating) => ratingMutation.mutate(rating)}
      onShellToggle={() => shellMutation.mutate(!isShelled)}
    />
  );
}
