import { useState } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { MarketingAuthGate } from "@/components/auth/MarketingAuthGate";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import * as api from "@/lib/api";
import { getToken, setToken, useCurrentUser } from "@/lib/auth";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const queryClient = useQueryClient();
  const [token, setTokenState] = useState<string | null>(getToken);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const { data: currentUser } = useCurrentUser();

  const recommendationsQuery = useQuery({
    queryKey: ["recommendations"],
    queryFn: () => api.getRecommendations(20),
    enabled: !!token,
  });

  const staffPicksQuery = useQuery({
    queryKey: ["staffPicks"],
    queryFn: () => api.getStaffPicks(6),
    enabled: !!token,
  });

  const activityQuery = useQuery({
    queryKey: ["activityFeed"],
    queryFn: () => api.getActivityFeed(10),
    enabled: !!token,
  });

  const myListsQuery = useQuery({
    queryKey: ["myLists"],
    queryFn: () => api.getMyLists(),
    enabled: !!token,
  });

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
      setTokenState(result.access_token);
      await queryClient.invalidateQueries();
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Authentication failed. Please try again.");
    } finally {
      setAuthLoading(false);
    }
  }

  if (!token) {
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
      currentUser={currentUser ?? undefined}
      userLists={myListsQuery.data ?? []}
      staffPicks={staffPicksQuery.data ?? []}
      activity={activityQuery.data ?? []}
      trendingLists={myListsQuery.data ?? []}
    />
  );
}
