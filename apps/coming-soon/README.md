# BookDB Coming Soon (Astro)

Ultra-minimal static landing page served via Cloudflare Pages.

## Structure

The Astro app lives at `apps/coming-soon` inside this repo. Astro builds the site into `dist`, which Cloudflare Pages can deploy as static assets. citeturn0search2turn0search4

## Local dev

```bash
cd apps/coming-soon
npm install
npm run dev
```

## Cloudflare Pages deploy (dashboard)

These settings assume a monorepo layout.

1. In Cloudflare, go to Workers & Pages and create a Pages project from this Git repo. citeturn0search3
2. Set build settings:
3. Root directory: `apps/coming-soon` (required for monorepos). citeturn0search1turn0search7
4. Build command: `npm run build`. citeturn0search2turn0search4
5. Build output directory: `dist`. citeturn0search2turn0search4

Cloudflare will build and deploy the static assets from `dist`.

## Custom domain

Use the Cloudflare dashboard to attach your domain to the Pages project.

1. Workers & Pages → select the project → Custom domains → Set up a domain. citeturn0search0
2. For an apex domain (example.com), the domain must be a Cloudflare zone and nameservers must point to Cloudflare. citeturn0search0
3. For a subdomain (www.example.com), add a CNAME to `<your-project>.pages.dev` after associating the domain in the Pages dashboard. citeturn0search0

Note: adding a CNAME without first associating the domain in the Pages UI can lead to resolution errors. citeturn0search0
