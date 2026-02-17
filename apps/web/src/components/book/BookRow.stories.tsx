import type { Meta, StoryObj } from "@storybook/react-vite";
import { BookRow } from "./BookRow";
import { Separator } from "@/components/ui/separator";
import { mockBooks, mockManyLists } from "@/lib/mockData";

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
    listOptions: mockManyLists,
  },
};

export const Shelled: Story = {
  args: {
    showActions: true,
    isShelled: true,
    listOptions: mockManyLists,
    selectedListIds: [mockManyLists[0].id, mockManyLists[5].id],
  },
};

export const InList: Story = {
  render: () => (
    <div className="max-w-xl">
      {mockBooks.slice(0, 5).map((book, i) => (
        <div key={book.id}>
          {i > 0 && <Separator />}
          <BookRow
            book={book}
            showActions
            listOptions={mockManyLists}
            selectedListIds={i % 2 === 0 ? [mockManyLists[0].id] : [mockManyLists[8].id]}
          />
        </div>
      ))}
    </div>
  ),
};

export const DiscoveryActions: Story = {
  args: {
    showActions: true,
    primaryAction: "amazon",
    listOptions: mockManyLists,
  },
};
