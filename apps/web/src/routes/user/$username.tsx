import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { UserPage } from "@/components/user/UserPage";
import * as api from "@/lib/api";
import { useCurrentUser } from "@/lib/auth";

export const Route = createFileRoute("/user/$username")({
  component: UserProfilePage,
});

function UserProfilePage() {
  const { username } = Route.useParams();
  const { data: me } = useCurrentUser();

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
    queryFn: () => api.getUserRatings(username, 3, "rating"),
  });

  if (userQuery.isLoading) {
    return <div className="py-8 text-muted-foreground text-sm">Loading…</div>;
  }

  if (userQuery.isError || !userQuery.data) {
    return <div className="py-8 text-destructive text-sm">User not found.</div>;
  }

  const user = userQuery.data;
  const isOwnProfile = me?.handle === username;
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
