import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { UserPage } from "@/components/user/UserPage";
import * as api from "@/lib/api";
import { useCurrentUser } from "@/lib/auth";

export const Route = createFileRoute("/users/$userId")({
  component: UserProfilePage,
});

function UserProfilePage() {
  const { userId } = Route.useParams();
  const { data: me } = useCurrentUser();

  const userQuery = useQuery({
    queryKey: ["user", userId],
    queryFn: () => api.getUser(userId),
  });

  const ratingsQuery = useQuery({
    queryKey: ["userRatings", userId],
    queryFn: () => api.getUserRatings(userId, 50),
  });

  const listsQuery = useQuery({
    queryKey: ["userLists", userId],
    queryFn: () => api.getUserLists(userId),
  });

  const activityQuery = useQuery({
    queryKey: ["userActivity", userId],
    queryFn: () => api.getUserActivity(userId, 10),
  });

  const favoritesQuery = useQuery({
    queryKey: ["userFavorites", userId],
    queryFn: () => api.getUserRatings(userId, 3, "rating"),
  });

  if (userQuery.isLoading) {
    return <div className="py-8 text-muted-foreground text-sm">Loading…</div>;
  }

  if (userQuery.isError || !userQuery.data) {
    return <div className="py-8 text-destructive text-sm">User not found.</div>;
  }

  const user = userQuery.data;
  const isOwnProfile = me?.id === userId;
  const favorites = (favoritesQuery.data ?? []).map((r) => r.book);

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
    />
  );
}
