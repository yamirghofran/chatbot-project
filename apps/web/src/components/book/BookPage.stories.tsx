import type { Meta, StoryObj } from "@storybook/react-vite";
import { BookPage } from "./BookPage";
import {
  mockBooks,
  mockBookStats,
  mockBookActivity,
  mockReviews,
  mockUser,
} from "@/lib/mockData";

const meta = {
  title: "Pages/BookPage",
  component: BookPage,
  args: {
    book: mockBooks[0],
    relatedBooks: mockBooks.slice(2, 6),
  },
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof BookPage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithRating: Story = {
  args: {
    rating: 4,
  },
};

export const Loved: Story = {
  args: {
    rating: 4.5,
    isLoved: true,
  },
};

export const FullPage: Story = {
  args: {
    rating: 4.5,
    isLoved: true,
    stats: mockBookStats,
    friendActivity: mockBookActivity,
    reviews: mockReviews,
    currentUser: mockUser,
  },
};
