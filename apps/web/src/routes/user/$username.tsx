import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UserPage } from "@/components/user/UserPage";
import { CenteredLoading } from "@/components/ui/CenteredLoading";
import * as api from "@/lib/api";
import { useCurrentUser, clearToken } from "@/lib/auth";

export const Route = createFileRoute("/user/$username")({
  component: UserProfilePage,
});

function UserProfilePage() {
  const { username } = Route.useParams();
  const { data: me } = useCurrentUser();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  function handleLogout() {
    clearToken();
    queryClient.setQueryData(["me"], null);
    navigate({ to: "/" });
  }

  const userQuery = useQuery({
    queryKey: ["user", username],
    queryFn: () => api.getUser(username),
  });

  const ratingsQuery = useQuery({
    queryKey: ["userRatings", username],
    queryFn: () => api.getUserRatings(username, 50),
  });

  const listsQuery = useQuery({
    queryKey: ["userLists", username],
    queryFn: () => api.getUserLists(username),
  });

  const activityQuery = useQuery({
    queryKey: ["userActivity", username],
    queryFn: () => api.getUserActivity(username, 10),
  });

  const favoritesQuery = useQuery({
    queryKey: ["userFavorites", username],
    queryFn: () => api.getUserFavorites(username, 3),
  });

  const createListMutation = useMutation({
    mutationFn: (name: string) => api.createList(name),
    onSuccess: async (created) => {
      await queryClient.invalidateQueries({ queryKey: ["myLists"] });
      await queryClient.invalidateQueries({ queryKey: ["userLists", username] });
      navigate({ to: "/lists/$listId", params: { listId: created.id } });
    },
  });

  function handleCreateList() {
    const name = window.prompt("List name");
    const trimmed = name?.trim();
    if (!trimmed) return;
    createListMutation.mutate(trimmed);
  }

  if (userQuery.isLoading) {
    return <CenteredLoading />;
  }

  if (userQuery.isError || !userQuery.data) {
    return <div className="py-8 text-destructive text-sm">User not found.</div>;
  }

  const user = userQuery.data;
  const isOwnProfile = me?.handle === username;
  const favorites = favoritesQuery.data ?? [];

  return (
    <UserPage
      user={user}
      isOwnProfile={isOwnProfile}
      favorites={favorites}
      ratedBooks={ratingsQuery.data ?? []}
      activity={activityQuery.data ?? []}
      lists={listsQuery.data ?? []}
      followingCount={0}
      followersCount={0}
      onCreateList={isOwnProfile ? handleCreateList : undefined}
      onLogout={isOwnProfile ? handleLogout : undefined}
    />
  );
}
