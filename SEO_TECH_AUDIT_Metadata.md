# Technical SEO Audit — Route Metadata (Next.js)

Scope: frontend/apps/commercial/app/* (Next.js App Router)
Date: 2025-11-06

## Findings (by cluster)

- **Global (app/layout.tsx)**
  - metadataBase set to https://rbee.dev
  - Global title/description/keywords, robots, OpenGraph/Twitter defined
  - Canonical: alternates.canonical = '/' (root) — OK
  - Note: Global keywords are largely ignored by Google; harmless but low impact

- **Home (app/page.tsx)**
  - export const metadata: alternates.canonical = '/'; description/keywords present; OG/Twitter present
  - JSON-LD added (Organization, WebSite) via structured-data utilities
  - Impact: Strong; recommend adding og:image (see Assets section)

- **Pricing (app/pricing/page.tsx)**
  - metadata: title/description/keywords, canonical '/pricing', OG/Twitter present
  - JSON-LD: Product/Offer schema injected (multiple SKUs)
  - Impact: High; add Product/Offer priceValidUntil updates automation; add breadcrumbs

- **Features hub (app/features/page.tsx)**
  - metadata: canonical '/features', OG/Twitter present
  - JSON-LD: FAQ schema injected (route-level)

- **Feature detail pages**
  - openai-compatible, multi-machine, heterogeneous-hardware, rhai-scripting, ssh-deployment, gdpr-compliance
  - Each has metadata: title/description/keywords, canonical, OG/Twitter
  - JSON-LD: FAQ schema injected (route-level) on all

- **Comparison pages**
  - /compare/rbee-vs-ollama, /rbee-vs-vllm, /rbee-vs-together-ai, /rbee-vs-ray-kserve
  - metadata: title/description/keywords, canonical, OG/Twitter
  - JSON-LD: FAQ schema injected (route-level)
  - Note: No “/compare” index route — add and include in sitemap

- **Personas & Industries**
  - Developers (route-level FAQ JSON-LD injected)
  - Enterprise (route-level FAQ JSON-LD injected)
  - Providers, Use Cases: metadata only (no route-level JSON-LD)
  - Industries (homelab, research, legal): metadata present; homelab adds FAQ JSON-LD; others metadata-only

- **Security / Legal**
  - Security: metadata only
  - Legal: Privacy/Terms metadata only

## Gaps / Risks

- **No per-route og:image definition**
  - No opengraph-image.ts or static OG assets configured per page
  - Action: Implement Next.js `opengraph-image.tsx` or static OG images for Home, Pricing, Features, Comparisons, and key personas

- **Inconsistent JSON-LD strategy**
  - Some pages inject FAQ via route; others via FAQTemplate(jsonLdEnabled)
  - Risk: potential duplication if both used on same page in future edits

- **Missing compare index route**
  - Provide /compare hub (metadata + index of all comparisons); add to sitemap

- **Global keywords**
  - Search engines largely ignore keywords; fine to keep but not a lever

## Recommended Actions

1) Add per-route OG images (static or dynamic) for primary pages
2) Standardize FAQ JSON-LD injection (choose route-level OR template-level per page, not both)
3) Add /compare (hub) with metadata + breadcrumb; link from Features/Pricing
4) Add BreadcrumbList JSON-LD on detail pages (Features, Comparisons)
5) Ensure canonical URLs match sitemap entries (they do in sampled routes)
