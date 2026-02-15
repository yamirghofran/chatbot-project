# BookDB Web

React 19 SPA built with Vite, TypeScript, TanStack Router, TanStack React Query, shadcn/ui, and Tailwind CSS v4. Component development and testing via Storybook.

## Prerequisites

- [Bun](https://bun.sh/) (package manager and runtime)
- [Node.js](https://nodejs.org/) >= 18

## Setup

```bash
# Clone the repository and navigate to the web app
cd apps/web

# Install dependencies
bun install

# Install Playwright browsers (required for browser tests)
bunx playwright install
```

## Development

```bash
# Start the Vite dev server with HMR
bun run dev
```

The app will be available at [http://localhost:5173](http://localhost:5173).

## Storybook

```bash
# Start Storybook dev server
bun run storybook
```

Storybook will be available at [http://localhost:6006](http://localhost:6006).

To build a static Storybook site:

```bash
bun run build-storybook
```

## Other Scripts

```bash
# Type-check and build for production
bun run build

# Preview the production build
bun run preview

# Run ESLint
bun run lint
```

## Project Structure

```
src/
├── components/ui/   # shadcn/ui components
├── lib/             # Utilities (cn helper, etc.)
├── routes/          # File-based routes (TanStack Router)
├── assets/          # Static assets
├── index.css        # Tailwind CSS + theme config
└── main.tsx         # App entry point
```
