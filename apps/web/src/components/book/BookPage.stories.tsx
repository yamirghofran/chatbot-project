import type { Meta, StoryObj } from "@storybook/react-vite";
import { BookPage } from "./BookPage";
import {
  mockBooks,
  mockBookStats,
  mockBookActivity,
  mockManyLists,
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

export const Shelled: Story = {
  args: {
    rating: 5,
    isShelled: true,
  },
};

export const LongDescription: Story = {
  args: {
    book: {
      ...mockBooks[0],
      description:
        "In a world teetering on the edge of transformation, this sweeping narrative follows three generations of a family bound together by secrets, ambition, and an ancient promise that refuses to stay buried. From the fog-laden streets of Victorian London to the sun-scorched plains of the American West, the story weaves together threads of love and betrayal, science and superstition, as each generation grapples with the consequences of choices made long before they were born. At its heart is a question that has haunted philosophers and poets alike: can we ever truly escape the past, or are we destined to repeat it? With prose that is by turns lyrical and devastating, this novel builds to a climax that will leave readers breathless and forever changed.",
    },
  },
};

export const FullPage: Story = {
  args: {
    rating: 5,
    isShelled: true,
    listOptions: mockManyLists,
    selectedListIds: [mockManyLists[0].id, mockManyLists[4].id],
    stats: mockBookStats,
    friendActivity: mockBookActivity,
    reviews: mockReviews,
    currentUser: mockUser,
  },
};
