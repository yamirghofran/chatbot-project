import type { Meta, StoryObj } from "@storybook/react-vite";
import { UserPage } from "./UserPage";
import { mockUser, mockBooks, mockLists } from "@/lib/mockData";

const meta = {
  title: "Pages/UserPage",
  component: UserPage,
  args: {
    user: mockUser,
    loved: mockBooks.slice(0, 6),
    lists: mockLists,
    followingCount: 42,
    followersCount: 128,
  },
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof UserPage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const EmptyLoved: Story = {
  args: {
    loved: [],
  },
};
