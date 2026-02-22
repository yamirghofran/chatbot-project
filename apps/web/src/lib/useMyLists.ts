import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "./api";
import { getToken } from "./auth";
import type { List } from "./types";

/**
 * Hook that manages the current user's lists for a specific book.
 * Provides lists, selected list IDs, and mutation helpers.
 */
export function useMyLists(bookId: string) {
  const queryClient = useQueryClient();
  const token = getToken();

  const { data: lists = [] } = useQuery<List[]>({
    queryKey: ["myLists"],
    queryFn: () => api.getMyLists(),
    enabled: !!token,
  });

  const selectedListIds = lists
    .filter((list) => list.books.some((b) => b.id === bookId))
    .map((list) => list.id);

  const addMutation = useMutation({
    mutationFn: ({ listId }: { listId: string }) =>
      api.addBookToList(listId, bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myLists"] }),
  });

  const removeMutation = useMutation({
    mutationFn: ({ listId }: { listId: string }) =>
      api.removeBookFromList(listId, bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myLists"] }),
  });

  const createMutation = useMutation({
    mutationFn: async (name: string) => {
      const result = await api.createList(name);
      if (bookId) {
        await api.addBookToList(result.id, bookId);
      }
      return result;
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["myLists"] }),
  });

  function toggleBook(listId: string, nextSelected: boolean) {
    if (nextSelected) {
      addMutation.mutate({ listId });
    } else {
      removeMutation.mutate({ listId });
    }
  }

  function createListForBook(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    createMutation.mutate(trimmed);
  }

  return { lists, selectedListIds, toggleBook, createListForBook };
}
