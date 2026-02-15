import type { Meta, StoryObj } from "@storybook/react-vite";
import { DiscoveryPage } from "./DiscoveryPage";
import {
  mockBooks,
  mockStaffPicks,
  mockActivity,
  mockLists,
} from "@/lib/mockData";

const meta = {
  title: "Pages/DiscoveryPage",
  component: DiscoveryPage,
  args: {
    books: mockBooks,
    staffPicks: mockStaffPicks,
    activity: mockActivity,
    trendingLists: mockLists,
  },
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof DiscoveryPage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
