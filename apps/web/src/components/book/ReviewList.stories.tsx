import type { Meta, StoryObj } from "@storybook/react-vite";
import { ReviewList } from "./ReviewList";
import { mockReviews, mockUser } from "@/lib/mockData";

const meta = {
  title: "Components/ReviewList",
  component: ReviewList,
  parameters: {
    layout: "centered",
  },
  decorators: [
    (Story) => (
      <div className="w-[600px]">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ReviewList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    reviews: mockReviews,
    currentUser: mockUser,
  },
};

export const Empty: Story = {
  args: {
    reviews: [],
    currentUser: mockUser,
  },
};
