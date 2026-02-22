import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { RatingPicker } from "./RatingPicker";
import { AmazonIcon } from "./AmazonIcon";
import { ShellButton } from "./ShellButton";
import { ThumbsUpIcon } from "./ThumbsUpIcon";
import { TurtleShellIcon } from "./TurtleShellIcon";

function InteractiveShellButton() {
  const [isShelled, setIsShelled] = useState(false);

  return (
    <ShellButton
      isShelled={isShelled}
      onClick={() => setIsShelled((prev) => !prev)}
    />
  );
}

function InteractiveRatingPicker() {
  const [rating, setRating] = useState<number | undefined>(3);

  return <RatingPicker value={rating} onChange={setRating} />;
}

const meta = {
  title: "UI/Icons",
} satisfies Meta;

export default meta;

type Story = StoryObj<typeof meta>;

export const TurtleShell: Story = {
  render: () => (
    <div className="flex items-center gap-6">
      <div className="flex flex-col items-center gap-2">
        <TurtleShellIcon className="size-10 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Default</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <TurtleShellIcon filled className="size-10 text-primary" />
        <span className="text-xs text-muted-foreground">Filled</span>
      </div>
    </div>
  ),
};

export const ShellActionButton: Story = {
  render: () => <InteractiveShellButton />,
};

export const Ratings: Story = {
  render: () => <InteractiveRatingPicker />,
};

export const ThumbsUp: Story = {
  render: () => <ThumbsUpIcon className="size-8 text-primary" />,
};

export const Amazon: Story = {
  render: () => <AmazonIcon className="size-8 text-primary" />,
};
