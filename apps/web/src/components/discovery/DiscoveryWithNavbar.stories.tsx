import type { Meta, StoryObj } from "@storybook/react-vite";
import { Navbar } from "@/components/navigation/Navbar";
import { DiscoveryPage } from "@/components/discovery/DiscoveryPage";
import {
  mockActivity,
  mockBooks,
  mockManyLists,
  mockLists,
  mockStaffPicks,
  mockUser,
} from "@/lib/mockData";

const meta = {
  title: "Pages/DiscoveryWithNavbar",
  parameters: {
    layout: "fullscreen",
    withContainer: false,
  },
  render: () => (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar user={mockUser} />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <DiscoveryPage
          books={mockBooks}
          userLists={mockManyLists}
          staffPicks={mockStaffPicks}
          activity={mockActivity}
          trendingLists={mockLists}
        />
      </main>
    </div>
  ),
} satisfies Meta;

export default meta;

export const Default: StoryObj = {};
