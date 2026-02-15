import type { Meta, StoryObj } from "@storybook/react-vite";
import { Navbar } from "./Navbar";
import { mockUser } from "@/lib/mockData";

const meta = {
  title: "Components/Navigation/Navbar",
  component: Navbar,
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof Navbar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const SignedIn: Story = {
  args: {
    user: mockUser,
  },
};

export const SignedOut: Story = {
  args: {
    user: undefined,
  },
};

export const CustomBrand: Story = {
  args: {
    brand: "BookDB Studio",
    user: mockUser,
  },
};
