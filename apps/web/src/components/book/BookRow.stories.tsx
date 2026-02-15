import type { Meta, StoryObj } from "@storybook/react-vite";
import { BookRow } from "./BookRow";
import { Separator } from "@/components/ui/separator";
import { mockBooks } from "@/lib/mockData";

const meta = {
  title: "Book/BookRow",
  component: BookRow,
  args: {
    book: mockBooks[0],
  },
} satisfies Meta<typeof BookRow>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: "compact",
  },
};

export const WithActions: Story = {
  args: {
    showActions: true,
  },
};

export const Favorited: Story = {
  args: {
    showActions: true,
    isFavorited: true,
  },
};

export const InList: Story = {
  render: () => (
    <div className="max-w-xl">
      {mockBooks.slice(0, 5).map((book, i) => (
        <div key={book.id}>
          {i > 0 && <Separator />}
          <BookRow book={book} showActions />
        </div>
      ))}
    </div>
  ),
};
