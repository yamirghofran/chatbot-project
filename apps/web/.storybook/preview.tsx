import React from "react";
import type { Preview } from "@storybook/react-vite";
import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../src/styles/globals.css";

function StoryProviders({ children }: { children: React.ReactNode }) {
  const queryClient = React.useMemo(() => new QueryClient(), []);
  const router = React.useMemo(() => {
    const rootRoute = createRootRoute({ component: () => <>{children}</> });
    return createRouter({
      routeTree: rootRoute,
      history: createMemoryHistory({ initialEntries: ["/"] }),
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

const preview: Preview = {
  decorators: [
    (Story, context) => {
      const withContainer = context.parameters?.withContainer !== false;

      return (
        <StoryProviders>
          <div className="min-h-dvh bg-background text-foreground">
            {withContainer ? (
              <main className="mx-auto max-w-5xl px-4 py-8">
                <Story />
              </main>
            ) : (
              <Story />
            )}
          </div>
        </StoryProviders>
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
      test: "todo",
    },
  },
};

export default preview;
