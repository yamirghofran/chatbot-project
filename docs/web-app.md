---
icon: lucide/monitor
---

# Web App Documentation

BookDB's web application is a modern React 19 single-page application built with TypeScript, featuring file-based routing, reactive data fetching, and a component library based on shadcn/ui.

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | React 19 |
| Build Tool | Vite 6 |
| Language | TypeScript |
| Routing | TanStack Router (file-based) |
| State Management | TanStack React Query |
| UI Components | shadcn/ui |
| Styling | Tailwind CSS v4 |
| Package Manager | Bun |
| Testing | Vitest, Playwright, Storybook |

## Project Structure

```
apps/web/
├── src/
│   ├── components/ui/    # shadcn/ui components
│   ├── lib/              # Utilities (cn helper, etc.)
│   ├── routes/           # File-based routes (TanStack Router)
│   ├── assets/           # Static assets
│   ├── styles/
│   │   └── globals.css   # Tailwind CSS + theme config
│   └── main.tsx          # App entry point
├── public/               # Static files
├── .storybook/           # Storybook configuration
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
└── package.json          # Dependencies
```

## Development

```bash
# Navigate to web app
cd apps/web

# Install dependencies
bun install

# Start development server
bun run dev

# Start Storybook
bun run storybook
```

The dev server runs at `http://localhost:5173` by default.

Storybook runs at `http://localhost:6006`.

## Build & Deploy

```bash
# Type-check and build for production
bun run build

# Preview production build locally
bun run preview

# Build static Storybook site
bun run build-storybook
```

## Routes

File-based routing via TanStack Router:

| Route | Description |
|-------|-------------|
| `/` | Home page |
| `/books/$bookId` | Book detail page |

## API Integration

The web app communicates with the FastAPI backend:

- Development: Configure proxy in `vite.config.ts` or set API base URL
- Production: Set `VITE_API_URL` environment variable

## Component Development

### Storybook

Storybook is configured for isolated component development:

```bash
bun run storybook
```

Stories are co-located with components or in `.stories.tsx` files.

### shadcn/ui

Add new components via shadcn CLI:

```bash
bunx shadcn add button
bunx shadcn add card
```

## Styling

Tailwind CSS v4 with CSS-first configuration in `globals.css`:

```css
@import "tailwindcss";

@theme {
  --color-primary: ...;
  --font-sans: ...;
}
```

Use the `cn()` utility for conditional class merging:

```tsx
import { cn } from "@/lib/utils";

<div className={cn("base-class", conditional && "conditional-class")} />
```

## Testing

```bash
# Run Vitest unit tests
bun run test

# Run Playwright e2e tests
bunx playwright test
```

## Coming Soon Page

A minimal static landing page built with Astro is available at `apps/coming-soon/`:

```bash
cd apps/coming-soon
npm install
npm run dev    # Local development
npm run build  # Build for deployment
```

Deployed to Cloudflare Pages for the pre-launch phase.
