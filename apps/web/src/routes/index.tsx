import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { MarketingAuthGate } from "@/components/auth/MarketingAuthGate";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import {
  mockBooks,
  mockStaffPicks,
  mockActivity,
  mockManyLists,
  mockLists,
} from "@/lib/mockData";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  if (!isAuthenticated) {
    return <MarketingAuthGate onAuthenticated={() => setIsAuthenticated(true)} />;
  }

  return (
    <DiscoveryPage
      books={mockBooks}
      userLists={mockManyLists}
      staffPicks={mockStaffPicks}
      activity={mockActivity}
      trendingLists={mockLists}
    />
  );
}
