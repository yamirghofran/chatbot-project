import { useState } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { MarketingAuthGate } from "@/components/auth/MarketingAuthGate";
import { CenteredLoading } from "@/components/ui/CenteredLoading";
import { Button } from "@/components/ui/button";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import * as api from "@/lib/api";
import { getToken, setToken, useCurrentUser } from "@/lib/auth";
import { homeStaffPicks, homeTrendingLists } from "@/lib/homePageMocks";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const queryClient = useQueryClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);
  const {
    data: currentUser,
    isLoading: meLoading,
    isError: meError,
    error: meErrorValue,
    refetch: refetchCurrentUser,
  } = useCurrentUser();

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

  async function handleAuthenticated({ email: e, password: p }: { email: string; password: string }) {
    setAuthLoading(true);
    setAuthError(null);
    try {
      const result = await api.login(e, p);
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
        email={email}
        onEmailChange={setEmail}
        password={password}
        onPasswordChange={setPassword}
      />
    );
  }

  if (meLoading) return <CenteredLoading />;

  if (hasToken && meError) {
    return (
      <div className="py-8">
        <p className="text-sm text-destructive">
          {meErrorValue instanceof Error
            ? meErrorValue.message
            : "Could not validate your session. Please try again."}
        </p>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="mt-3"
          onClick={() => refetchCurrentUser()}
        >
          Retry
        </Button>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <MarketingAuthGate
        onAuthenticated={handleAuthenticated}
        error={authError}
        isLoading={authLoading}
        email={email}
        onEmailChange={setEmail}
        password={password}
        onPasswordChange={setPassword}
      />
    );
  }

  return (
    <DiscoveryPage
      books={recommendationsQuery.data ?? []}
      booksLoading={recommendationsQuery.isLoading}
      currentUser={currentUser}
      userLists={myListsQuery.data ?? []}
      staffPicks={resolvedStaffPicks}
      activity={activityQuery.data ?? []}
      trendingLists={resolvedTrendingLists}
    />
  );
}
