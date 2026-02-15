import type { Preview } from "@storybook/react-vite";
import "../src/styles/globals.css";

const preview: Preview = {
  decorators: [
    (Story, context) => {
      const withContainer = context.parameters?.withContainer !== false;

      return (
        <div className="min-h-dvh bg-background text-foreground">
          {withContainer ? (
            <main className="mx-auto max-w-5xl px-4 py-8">
              <Story />
            </main>
          ) : (
            <Story />
          )}
        </div>
      );
    },
  ],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },

    a11y: {
      // 'todo' - show a11y violations in the test UI only
      // 'error' - fail CI on a11y violations
      // 'off' - skip a11y checks entirely
      test: "todo",
    },
  },
};

export default preview;
