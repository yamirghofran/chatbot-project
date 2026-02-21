import { useState } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { MarketingAuthGate } from "@/components/auth/MarketingAuthGate";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import * as api from "@/lib/api";
import { getToken, setToken, useCurrentUser } from "@/lib/auth";
import { homeStaffPicks, homeTrendingLists } from "@/lib/homePageMocks";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const queryClient = useQueryClient();
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const { data: currentUser, isLoading: meLoading } = useCurrentUser();

  const hasToken = !!getToken();

  const recommendationsQuery = useQuery({
    queryKey: ["recommendations"],
    queryFn: () => api.getRecommendations(20),
    enabled: !!currentUser,
  });

  const activityQuery = useQuery({
    queryKey: ["activityFeed"],
    queryFn: () => api.getActivityFeed(10),
    enabled: !!currentUser,
  });

  const myListsQuery = useQuery({
    queryKey: ["myLists"],
    queryFn: () => api.getMyLists(),
    enabled: !!currentUser,
  });
  const teamListsQuery = useQuery({
    queryKey: ["userLists", "bookdb"],
    queryFn: () => api.getUserLists("bookdb"),
    enabled: !!currentUser,
  });

  const resolvedTrendingLists = teamListsQuery.data ?? homeTrendingLists;
  const resolvedStaffPicks = homeStaffPicks;

  async function handleAuthenticated({
    email,
    password,
    mode,
    name,
    username,
  }: {
    email: string;
    password: string;
    mode: "signin" | "signup";
    name?: string;
    username?: string;
  }) {
    setAuthLoading(true);
    setAuthError(null);
    try {
      let result: api.TokenResponse;
      if (mode === "signup" && name && username) {
        result = await api.register(email, password, name, username);
      } else {
        result = await api.login(email, password);
      }
      setToken(result.access_token);
      await queryClient.invalidateQueries();
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Authentication failed. Please try again.");
    } finally {
      setAuthLoading(false);
    }
  }

  if (!hasToken) {
    return (
      <MarketingAuthGate
        onAuthenticated={handleAuthenticated}
        error={authError}
        isLoading={authLoading}
      />
    );
  }

  if (meLoading) return null;

  if (!currentUser) {
    return (
      <MarketingAuthGate
        onAuthenticated={handleAuthenticated}
        error={authError}
        isLoading={authLoading}
      />
    );
  }

  return (
    <DiscoveryPage
      books={recommendationsQuery.data ?? []}
      currentUser={currentUser}
      userLists={myListsQuery.data ?? []}
      staffPicks={resolvedStaffPicks}
      activity={activityQuery.data ?? []}
      trendingLists={resolvedTrendingLists}
    />
  );
}
