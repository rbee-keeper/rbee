# Technical SEO Audit — Structured Data (JSON-LD)

Scope: frontend/apps/commercial (App Router)
Date: 2025-11-06

## Central utilities
- lib/seo/structured-data.ts provides:
  - Organization, WebSite schemas
  - Product/Offer schemas (Pricing SKUs)
  - FAQPage schemas (feature and comparison keys)
  - BreadcrumbList schema
  - ComparisonTable (custom) schema
  - generateJSONLD helper

## Route-level JSON-LD injection (confirmed)
- Home (/): Organization + WebSite (in app/page.tsx)
- Pricing (/pricing): Product/Offer for 4 SKUs (in app/pricing/page.tsx)
- Features hub (/features): FAQ schema
- Feature detail pages:
  - /features/openai-compatible: FAQ
  - /features/multi-machine: FAQ
  - /features/heterogeneous-hardware: FAQ
  - /features/rhai-scripting: FAQ
  - /features/ssh-deployment: FAQ
  - /features/gdpr-compliance: FAQ (plus detail template content)
- Comparisons (/compare/*): FAQ for each vs page (Ollama, vLLM, Together.ai, Ray+KServe)
- Personas/Industries with route-level FAQ:
  - /developers: FAQ
  - /enterprise: FAQ
  - /industries/homelab: FAQ

## Template-level JSON-LD injection
- FAQTemplate supports `jsonLdEnabled` to inline FAQPage schema inside components.
- Examples observed in PageProps:
  - Pricing FAQ: `jsonLdEnabled: true`
  - Privacy FAQ: `jsonLdEnabled: true`
  - Community/Legal/Security FAQ templates do not consistently enable JSON-LD (varies by page)

## Duplication risk
- Some routes inject FAQ via the route file, others via the template. If both are enabled on the same page in future edits, you could ship duplicate FAQ schemas.

## Missing / underused schemas
- BreadcrumbList: Utility exists but not used on detail routes (features/comparisons). Recommend adding BreadcrumbList on:
  - Features hub and feature details
  - Compare hub (when created) and each comparison page
- Comparison schema: `getComparisonSchema` is defined but not used. Consider adding to vs pages (with caution; "ComparisonTable" is not a standard Schema.org type — consider using a Product/Service schema with properties, or an Article with table markup and FAQ schema instead).
- Article/HowTo: Install quickstart could use HowTo schema on a dedicated "Getting Started" page.

## Coverage matrix (high level)
- Organization/WebSite: Home ✅
- Product/Offer: Pricing ✅
- FAQPage: Features hub ✅, Feature details ✅, Comparisons ✅, Developers/Enterprise/Homelab ✅, Pricing FAQ (via template) ✅, Privacy FAQ (via template) ✅
- BreadcrumbList: ❌ (not implemented)
- Compare hub route: ❌ (not present)

## Recommendations
1) Pick ONE pattern for FAQs per page:
   - Prefer template-level `jsonLdEnabled` for sections that render FAQs; avoid duplicating FAQ at route level on the same page.
2) Add BreadcrumbList JSON-LD to feature details and comparison pages using `getBreadcrumbSchema`.
3) Use Product/Offer only on Pricing (avoid elsewhere).
4) Keep Organization/WebSite to Home only (or sitewide via RootLayout once; but avoid multiple injections).
5) Add a /compare hub and include an index JSON-LD (BreadcrumbList + FAQ about comparisons).
6) Avoid "ComparisonTable" custom type; stick to schema.org types (FAQPage, Article) for reliability.
