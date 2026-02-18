import type { Meta, StoryObj } from "@storybook/react-vite";
import { ListPage } from "./ListPage";
import { mockLists, mockUser } from "@/lib/mockData";

const meta = {
  title: "Pages/ListPage",
  component: ListPage,
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof ListPage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    list: mockLists[0],
  },
};

export const NoDescription: Story = {
  args: {
    list: mockLists[2],
  },
};

export const EditMode: Story = {
  args: {
    list: mockLists[0],
    isOwner: true,
  },
};

export const GridView: Story = {
  args: {
    list: mockLists[1],
    defaultViewMode: "grid",
  },
};

export const EmptyList: Story = {
  args: {
    list: {
      id: "l-empty",
      name: "Books to Find",
      description: "A list waiting to be filled.",
      owner: mockUser,
      books: [],
    },
    isOwner: true,
  },
};
