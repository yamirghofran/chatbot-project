import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BookPage } from "@/components/book/BookPage";
import { CenteredLoading } from "@/components/ui/CenteredLoading";
import * as api from "@/lib/api";
import { useMyLists } from "@/lib/useMyLists";
import { useCurrentUser } from "@/lib/auth";
import type { Book } from "@/lib/types";

export const Route = createFileRoute("/books/$bookId")({
  component: BookDetailPage,
});

function BookDetailPage() {
  const { bookId } = Route.useParams();
  const queryClient = useQueryClient();
  const { data: me } = useCurrentUser();
  const myRatingQueryKey = ["myRating", bookId];

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

  const myRatingQuery = useQuery({
    queryKey: myRatingQueryKey,
    queryFn: () => api.getMyRating(bookId),
    enabled: !!me,
  });

  const { lists, selectedListIds, toggleBook, createListForBook } = useMyLists(bookId);

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
    return <CenteredLoading />;
  }

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
