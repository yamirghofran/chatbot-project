import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BookPage } from "@/components/book/BookPage";
import * as api from "@/lib/api";
import { useMyLists } from "@/lib/useMyLists";

export const Route = createFileRoute("/books/$bookId")({
  component: BookDetailPage,
});

function BookDetailPage() {
  const { bookId } = Route.useParams();
  const queryClient = useQueryClient();
  const token = getToken();

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

  const { lists, selectedListIds, toggleBook, createListForBook } = useMyLists(bookId);

  const ratingMutation = useMutation({
    mutationFn: (rating: number | undefined) =>
      rating !== undefined
        ? api.upsertRating(bookId, rating)
        : api.deleteRating(bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["book", bookId] }),
  });

  const shellMutation = useMutation({
    mutationFn: (isShelled: boolean) =>
      isShelled ? api.addToShell(bookId) : api.removeFromShell(bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myShell"] }),
  });

  if (bookQuery.isLoading) {
    return <div className="py-8 text-muted-foreground text-sm">Loading…</div>;
  }

  if (bookQuery.isError || !bookQuery.data) {
    return <div className="py-8 text-destructive text-sm">Book not found.</div>;
  }

  const book = bookQuery.data;
  const stats = book.stats ?? undefined;

  return (
    <BookPage
      book={book}
      stats={stats}
      relatedBooks={relatedQuery.data ?? []}
      reviews={reviewsQuery.data ?? []}
      listOptions={lists}
      selectedListIds={selectedListIds}
      onToggleList={(listId, nextSelected) => toggleBook(listId, nextSelected)}
      onCreateList={(name) => createListForBook(name)}
      onRatingChange={(rating) => ratingMutation.mutate(rating)}
      onShellToggle={() => shellMutation.mutate(true)}
    />
  );
}
