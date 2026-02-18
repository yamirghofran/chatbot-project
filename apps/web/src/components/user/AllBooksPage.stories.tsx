import type { Meta, StoryObj } from "@storybook/react-vite";
import { AllBooksPage } from "./AllBooksPage";
import { mockRatedBooks } from "@/lib/mockData";

const meta = {
  title: "Pages/AllBooksPage",
  component: AllBooksPage,
  args: {
    ratedBooks: mockRatedBooks,
  },
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof AllBooksPage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Empty: Story = {
  args: {
    ratedBooks: [],
  },
};
