import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ListPage } from "@/components/list/ListPage";
import * as api from "@/lib/api";
import { useCurrentUser } from "@/lib/auth";

export const Route = createFileRoute("/lists/$listId")({
  component: ListDetailPage,
});

function ListDetailPage() {
  const { listId } = Route.useParams();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { data: me } = useCurrentUser();

  const listQuery = useQuery({
    queryKey: ["list", listId],
    queryFn: () => api.getList(listId),
  });

  const updateMutation = useMutation({
    mutationFn: (body: { name?: string; description?: string }) =>
      api.updateList(listId, body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["list", listId] }),
  });

  const removeBookMutation = useMutation({
    mutationFn: (bookId: string) => api.removeBookFromList(listId, bookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["list", listId] }),
  });

  const reorderMutation = useMutation({
    mutationFn: (bookIds: string[]) => api.reorderList(listId, bookIds),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["list", listId] }),
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteList(listId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["myLists"] });
      navigate({ to: "/" });
    },
  });

  if (listQuery.isLoading) {
    return <div className="py-8 text-muted-foreground text-sm">Loading…</div>;
  }

  if (listQuery.isError || !listQuery.data) {
    return <div className="py-8 text-destructive text-sm">List not found.</div>;
  }

  const list = listQuery.data;
  const isOwner = me?.id === list.owner.id;

  return (
    <ListPage
      list={list}
      isOwner={isOwner}
      onUpdateName={(name) => updateMutation.mutate({ name })}
      onUpdateDescription={(description) => updateMutation.mutate({ description })}
      onRemoveBook={(bookId) => removeBookMutation.mutate(bookId)}
      onReorder={(bookIds) => reorderMutation.mutate(bookIds)}
      onDeleteList={() => deleteMutation.mutate()}
    />
  );
}
