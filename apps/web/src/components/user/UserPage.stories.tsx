import type { Meta, StoryObj } from "@storybook/react-vite";
import { UserPage } from "./UserPage";
import {
  mockUser,
  mockFavorites,
  mockRatedBooks,
  mockUserActivity,
  mockLists,
} from "@/lib/mockData";

const meta = {
  title: "Pages/UserPage",
  component: UserPage,
  args: {
    user: mockUser,
    favorites: mockFavorites,
    ratedBooks: mockRatedBooks,
    activity: mockUserActivity,
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

export const OwnProfile: Story = {
  args: {
    isOwnProfile: true,
  },
};

export const EmptyProfile: Story = {
  args: {
    favorites: [],
    ratedBooks: [],
    activity: [],
    lists: [],
    followingCount: 0,
    followersCount: 0,
  },
};

export const NewUser: Story = {
  args: {
    user: {
      id: "u99",
      handle: "newuser",
      displayName: "New User",
      avatarUrl: undefined,
    },
    isOwnProfile: true,
    favorites: [],
    ratedBooks: [],
    activity: [],
    lists: [],
    followingCount: 0,
    followersCount: 0,
  },
};
