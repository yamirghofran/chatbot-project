import { createFileRoute } from "@tanstack/react-router";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import {
  mockBooks,
  mockStaffPicks,
  mockActivity,
  mockLists,
} from "@/lib/mockData";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return (
    <DiscoveryPage
      books={mockBooks}
      staffPicks={mockStaffPicks}
      activity={mockActivity}
      trendingLists={mockLists}
    />
  );
}
