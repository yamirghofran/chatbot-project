import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { MarketingAuthGate } from "./MarketingAuthGate";

const meta = {
  title: "Pages/MarketingAuthGate",
  component: MarketingAuthGate,
  parameters: {
    layout: "fullscreen",
  },
  render: () => (
    <div className="min-h-dvh bg-background text-foreground">
      <main className="mx-auto max-w-5xl px-4 py-8">
        <MarketingAuthGate />
      </main>
    </div>
  ),
} satisfies Meta<typeof MarketingAuthGate>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Submitted: Story = {
  render: () => {
    const [email, setEmail] = useState<string | null>(null);

    return (
      <div className="min-h-dvh bg-background text-foreground">
        <main className="mx-auto max-w-5xl px-4 py-8">
          {!email ? (
            <MarketingAuthGate
              onAuthenticated={(payload) => {
                setEmail(payload.email);
              }}
            />
          ) : (
            <section className="rounded-xl bg-card p-6 sm:p-8">
              <p className="text-sm text-muted-foreground">Signed in as {email}</p>
            </section>
          )}
        </main>
      </div>
    );
  },
};
