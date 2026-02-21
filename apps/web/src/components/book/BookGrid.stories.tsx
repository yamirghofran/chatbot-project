import type { Meta, StoryObj } from "@storybook/react-vite";
import { BookGrid } from "./BookGrid";
import { mockBooks } from "@/lib/mockData";

const meta = {
  title: "Book/BookGrid",
  component: BookGrid,
  args: {
    books: mockBooks,
    columns: 5,
  },
} satisfies Meta<typeof BookGrid>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const FourColumns: Story = {
  args: { columns: 4 },
};

export const ThreeColumns: Story = {
  args: { columns: 3, books: mockBooks.slice(0, 6) },
};

export const FewBooks: Story = {
  args: { books: mockBooks.slice(0, 4), columns: 5 },
};
