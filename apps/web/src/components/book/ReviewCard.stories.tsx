import type { Meta, StoryObj } from "@storybook/react-vite";
import { ReviewCard } from "./ReviewCard";
import { mockReviews } from "@/lib/mockData";

const meta = {
  title: "Components/ReviewCard",
  component: ReviewCard,
  parameters: {
    layout: "centered",
  },
  decorators: [
    (Story) => (
      <div className="w-[500px]">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ReviewCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    review: { ...mockReviews[1], isLikedByMe: false },
  },
};

export const Liked: Story = {
  args: {
    review: mockReviews[0],
  },
};

export const AsReply: Story = {
  args: {
    review: mockReviews[0].replies![0],
    isReply: true,
  },
};
