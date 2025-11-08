# Marketplace: Structure & Purpose

**Analysis Date:** 2025-11-06  
**Scope:** `frontend/apps/marketplace`  
**Purpose:** Complete marketplace architecture analysis

---

## Route Structure

**Total Routes:** 6 page.tsx files  
**Domain:** marketplace.rbee.ai (separate from rbee.dev)

### Main Routes

#### `/` (Marketplace Home)
**File:** `app/page.tsx`  
**Purpose:** Marketplace landing/hero page  
**Data:** Static content only (no API calls)  

**Content:**
- Hero section: "AI Model Marketplace"
- Tagline: "Discover and explore state-of-the-art language models from HuggingFace"
- CTA: "Browse Models" → `/models`
- 3 feature cards:
  1. Lightning Fast (SSG pre-rendering)
  2. SEO Optimized (semantic HTML, structured data)
  3. Rich Metadata (downloads, likes, tags)

**SEO Focus:** Emphasizes "Pre-rendered for blazing-fast performance and maximum SEO"

---

#### `/models` (Model List)
**File:** `app/models/page.tsx`  
**Purpose:** Browse all models (top 100)  
**Data:** HuggingFace API via `fetchTopModels(100)` at build time  

**Rendering:** Pure SSG (Static Site Generation)  
- No client-side JavaScript for content
- All data pre-rendered at build time
- Static HTML table output

**Content:**
- Model table with columns: Model, Author, Downloads, Likes, Tags
- Each row links to `/models/[slug]`

**SEO Strategy (from SEO_ARCHITECTURE.md):**
- ✅ All content in initial HTML
- ✅ No "loading..." states
- ✅ No hydration mismatches
- ✅ Perfect for Google/Bing crawlers
- ✅ Works in text browsers (Lynx/w3m)

---

#### `/models/[slug]` (Model Detail)
**File:** `app/models/[slug]/page.tsx`  
**Purpose:** Individual model detail pages  
**Data:** HuggingFace API via `getHuggingFaceModel(slug)` at build time  

**Rendering:** SSG with `generateStaticParams()`  
- Pre-renders top 100 models at build time
- Uses `ModelDetailWithInstall` wrapper (TEAM-421)

**Content (from TEAM-421 analysis):**
- Model name, author, description
- Stats: downloads, likes, size
- Model files (siblings)
- Compatible workers
- Metadata, config, chat template
- Tags, example prompts
- **InstallCTA** (Next.js only - conversion prompt)

**SEO Metadata:**
```typescript
export async function generateMetadata({ params }): Promise<Metadata> {
  const model = await getHuggingFaceModel(params.slug)
  return {
    title: `${model.name} - AI Model | rbee Marketplace`,
    description: model.description,
    // ... full OG/Twitter metadata
  }
}
```

**JSON-LD:** Not currently implemented (opportunity)

---

#### `/workers` (Worker List)
**File:** `app/workers/page.tsx`  
**Purpose:** Browse available workers  
**Data:** Static worker catalog (WORKERS object)  

**Content:**
- Worker cards/list
- Links to `/workers/[workerId]`

---

#### `/workers/[workerId]` (Worker Detail)
**File:** `app/workers/[workerId]/page.tsx`  
**Purpose:** Individual worker detail pages  
**Data:** Static WORKERS object (hardcoded)  

**Rendering:** SSG with `generateStaticParams()`  
- Pre-renders 4 workers: cpu-llm, cuda-llm, metal-llm, rocm-llm

**Content (from TEAM-421 refactor):**
- Worker name, description, version
- Type badge (CPU/CUDA/Metal/ROCm)
- Platform support
- Requirements
- Features
- **InstallCTA** (Next.js only - conversion prompt)

**SEO Metadata:**
```typescript
export async function generateMetadata({ params }): Promise<Metadata> {
  const worker = WORKERS[params.workerId]
  return {
    title: `${worker.name} v${worker.version} - Worker | rbee Marketplace`,
    description: worker.description,
    // ... full OG/Twitter metadata
  }
}
```

**Note:** TEAM-421 recently refactored this from 228 lines of inline JSX to 13 lines using `WorkerDetailWithInstall` wrapper

---

#### `/search` (Search Page)
**File:** `app/search/page.tsx`  
**Purpose:** Search functionality (likely future)  
**Data:** Unknown (not analyzed)

**Note:** Layout references search action in WebSite schema:
```typescript
potentialAction: {
  '@type': 'SearchAction',
  target: 'https://rbee.dev/search?q={search_term_string}',
}
```

---

## Navigation Layout

**Component:** `MarketplaceNav` (client component)  
**File:** `components/MarketplaceNav.tsx`

**Structure:**
- Fixed top navigation with backdrop blur
- 3-zone grid layout:
  - Zone A: rbee logo (links to `/`)
  - Zone B: Navigation links (Models, Datasets, Spaces)
  - Zone C: Actions (Docs, GitHub, Theme toggle, "Back to rbee.dev")

**Navigation Links:**
1. `/models` - Models (implemented)
2. `/datasets` - Datasets (NOT IMPLEMENTED - nav exists but no route)
3. `/spaces` - Spaces (NOT IMPLEMENTED - nav exists but no route)

**External Links:**
- Docs: `https://github.com/veighnsche/llama-orch/tree/main/docs`
- GitHub: `https://github.com/veighnsche/llama-orch`
- Back to rbee.dev: `https://rbee.dev`

**Accessibility:**
- Skip to content link
- ARIA labels and current page indicators
- Semantic nav element

---

## Footer

**Component:** `Footer` from `@rbee/ui/organisms`  
**Shared with:** Commercial site

**Content:** (Assumed based on shared component)
- Links to legal pages
- Social media links
- Copyright info

---

## Theme System

**Provider:** `ThemeProvider` from `next-themes`  
**Configuration:**
- `attribute="class"` - Uses class-based dark mode
- `defaultTheme="system"` - Respects OS preference
- `enableSystem` - Allows system theme detection
- `disableTransitionOnChange` - No flash on theme switch

**Theme Toggle:** Shared component from `@rbee/ui/molecules`

**CSS Loading (from layout.tsx):**
```typescript
import "./globals.css"      // App-specific CSS with JIT scanning
import "@rbee/ui/styles.css" // Pre-built design tokens + component styles
```

**Design Tokens:** Loaded from `@rbee/ui/styles.css` which imports `theme-tokens.css`

---

## Data Flow Architecture

### Models (HuggingFace)

**Build Time (SSG):**
```
HuggingFace API
    ↓
marketplace-sdk (WASM in Node.js)
    ↓
marketplace-node (Node.js wrapper)
    ↓
Next.js SSG (generateStaticParams)
    ↓
Static HTML files
```

**Runtime (Browser):**
```
Static HTML
    ↓
User clicks "Open in rbee App" (InstallCTA)
    ↓
Deep link: rbee://download-model/{id}
    ↓
Opens rbee-keeper (Tauri app)
```

**Note:** TEAM-421 implemented environment-aware actions:
- Tauri: Direct download via `invoke('model_download')`
- Next.js: Deep link to rbee-keeper

---

### Workers (Static Catalog)

**Build Time:**
```
WORKERS object (hardcoded in page.tsx)
    ↓
Next.js SSG
    ↓
Static HTML files
```

**Runtime:**
```
Static HTML
    ↓
User clicks "Open in rbee App" (InstallCTA)
    ↓
Deep link: rbee://install-worker/{id}
    ↓
Opens rbee-keeper (Tauri app)
```

---

## SSR/SSG Strategy

**From SEO_ARCHITECTURE.md:**

### Pure Static Site Generation (SSG)

**Benefits:**
- ✅ All content pre-rendered at build time
- ✅ Zero JavaScript needed for content display
- ✅ Perfect for search engine crawlers
- ✅ Instant page loads (no hydration delay)
- ✅ Works with JavaScript disabled

**Implementation:**
- Server components by default (no 'use client')
- Client components only for interactivity (nav, theme toggle, CTAs)
- No client-side data fetching for content
- No loading states or skeletons

**Example HTML Output:**
```html
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Author</th>
      <th>Downloads</th>
      <th>Likes</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gpt2</td>
      <td>openai-community</td>
      <td>11.5M</td>
      <td>3.0K</td>
      <td><span>transformers</span></td>
    </tr>
    <!-- 99 more rows... -->
  </tbody>
</table>
```

**Future: Server-Side Filtering**

SEO_ARCHITECTURE.md mentions potential for server-side filtering:
```typescript
export default async function ModelsPage({
  searchParams
}: {
  searchParams: { sort?: string; tag?: string; search?: string }
}) {
  const models = await fetchTopModels(100)
  
  // Server-side filtering (no client JS needed)
  let filtered = models
  if (searchParams.search) {
    filtered = filtered.filter(m => m.name.includes(searchParams.search!))
  }
  
  return <ModelTable models={filtered} />
}
```

**Note:** Currently not implemented, but architecture supports it

---

## Intended Story/Purpose

**From homepage and SEO_ARCHITECTURE.md:**

### Primary Value Proposition

"Discover models fast" - Emphasis on:
1. **Speed:** Pre-rendered SSG, instant loading
2. **SEO:** Maximum search visibility
3. **Discovery:** Browse HuggingFace models easily

### Target Audience

1. **Developers** looking for models to use with rbee
2. **Search engines** (Google, Bing) - SEO-first design
3. **rbee users** exploring available models/workers

### Conversion Funnel

```
User finds model via Google
    ↓
Lands on marketplace.rbee.ai/models/[slug]
    ↓
Sees InstallCTA: "Install rbee to download this model"
    ↓
Clicks "Download rbee" or "Open in rbee App"
    ↓
Converts to rbee user
```

**Key Insight:** Marketplace is a **lead generation tool** for rbee, not a standalone product

---

## Workers Description

**Current State:**

Workers are described as execution environments for models:
- **cpu-llm:** CPU-based LLM inference
- **cuda-llm:** CUDA GPU acceleration
- **metal-llm:** Apple Metal GPU acceleration
- **rocm-llm:** AMD ROCm GPU acceleration

**Worker Detail Pages Show:**
- Platform support (Linux, macOS, Windows)
- Requirements (hardware, dependencies)
- Features (capabilities)
- Version info
- Installation instructions (via InstallCTA)

**Location in Codebase:**
- Hardcoded WORKERS object in `/workers/[workerId]/page.tsx`
- Not fetched from API (unlike models)

**Future:** Could be expanded to show:
- Performance benchmarks
- Compatible models
- Configuration options
- Community ratings

---

## Missing Routes (Nav Links Exist)

### `/datasets`
**Status:** Navigation link exists, but NO ROUTE IMPLEMENTED  
**Purpose:** (Intended) Browse HuggingFace datasets  
**Priority:** Unknown

### `/spaces`
**Status:** Navigation link exists, but NO ROUTE IMPLEMENTED  
**Purpose:** (Intended) Browse HuggingFace Spaces  
**Priority:** Unknown

**Note:** These are standard HuggingFace categories, likely planned for future

---

## Recent Changes (TEAM-421)

**Environment-Aware Presentation Layer:**

1. **InstallCTA Component** (NEW)
   - Shows only in Next.js (hidden in Tauri)
   - Prompts: "Install rbee to download/install this model/worker"
   - Conversion-focused

2. **WorkerDetailWithInstall** (NEW)
   - Client wrapper for worker detail pages
   - Uses shared `ArtifactDetailPageTemplate`
   - Includes InstallCTA

3. **ModelDetailWithInstall** (UPDATED)
   - Now uses InstallCTA instead of custom install section
   - Consistent with worker pages

4. **Worker Page Refactor**
   - Before: 228 lines of inline JSX
   - After: 13 lines using wrapper
   - Removed 215 lines of duplicate code

**Result:** Consistent architecture across model and worker detail pages

---

## SEO Metadata Status

### Layout-Level Metadata

**File:** `app/layout.tsx`

```typescript
export const metadata: Metadata = {
  title: {
    default: "rbee Model Marketplace - AI Language Models",
    template: "%s | rbee Marketplace"
  },
  description: "Browse and discover AI language models...",
  keywords: ["AI", "language models", "LLM", "machine learning", "marketplace"],
  openGraph: { ... },
  twitter: { ... },
}
```

**Good:**
- ✅ Template pattern for page titles
- ✅ OG/Twitter cards configured
- ✅ Keywords defined

**Issues:**
- Generic description (could be more compelling)
- Limited keywords (only 5 terms)

---

### Page-Level Metadata

**Models Detail (`/models/[slug]`):**
- ✅ Dynamic metadata via `generateMetadata()`
- ✅ Custom title per model
- ✅ Custom description (from model data)
- ❌ No JSON-LD schemas
- ❌ No keywords per model

**Workers Detail (`/workers/[workerId]`):**
- ✅ Dynamic metadata via `generateMetadata()`
- ✅ Custom title per worker
- ✅ Custom description
- ❌ No JSON-LD schemas
- ❌ No keywords per worker

**Other Pages:**
- `/` (Home): Inherits layout metadata
- `/models` (List): Inherits layout metadata
- `/workers` (List): Inherits layout metadata
- `/search`: Unknown

**Coverage:** 2/6 pages have custom metadata (33%)

---

## Opportunities for Improvement

### 1. Add JSON-LD Schemas

**Model Detail Pages:**
```typescript
{
  "@type": "SoftwareApplication",
  "name": model.name,
  "description": model.description,
  "author": { "@type": "Person", "name": model.author },
  "downloadUrl": `rbee://download-model/${model.id}`,
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.5",
    "reviewCount": model.likes
  }
}
```

**Worker Detail Pages:**
```typescript
{
  "@type": "SoftwareApplication",
  "name": worker.name,
  "version": worker.version,
  "operatingSystem": worker.platforms.join(", "),
  "downloadUrl": `rbee://install-worker/${worker.id}`
}
```

---

### 2. Implement Missing Routes

- `/datasets` - HuggingFace datasets
- `/spaces` - HuggingFace Spaces
- Or remove nav links if not planned

---

### 3. Add Breadcrumbs

**Example:**
```
Home > Models > meta-llama/Llama-2-7b-chat-hf
```

With BreadcrumbList schema for rich snippets

---

### 4. Enhance List Pages

**Models List:**
- Add filtering (by tag, author, popularity)
- Add sorting (downloads, likes, recent)
- Add pagination or infinite scroll
- Add search

**Workers List:**
- Currently minimal
- Could add filtering by platform, type
- Could add comparison table

---

### 5. Add OG Images

**Dynamic OG image generation:**
- `/api/og/model/[slug]` - Model card images
- `/api/og/worker/[workerId]` - Worker card images

**Better social sharing on Twitter, Discord, LinkedIn**

---

## Summary

**Current State:**
- ✅ Clean SSG architecture (SEO-first)
- ✅ Models from HuggingFace (top 100)
- ✅ Workers catalog (4 types)
- ✅ Environment-aware CTAs (TEAM-421)
- ✅ Consistent presentation layer
- ✅ Theme system working
- ❌ Datasets/Spaces routes missing
- ❌ Limited SEO metadata (no JSON-LD)
- ❌ No filtering/search on list pages

**Intended Story:**
- "Discover models fast" - Speed + SEO focus
- Lead generation for rbee (conversion funnel)
- Showcase HuggingFace ecosystem integration

**Architecture Strengths:**
- Pure SSG (no client-side data fetching)
- Environment-aware actions (Tauri vs Next.js)
- Shared UI components with commercial site
- Consistent design tokens

**Next Steps:**
1. Add JSON-LD schemas to model/worker pages
2. Implement or remove datasets/spaces routes
3. Add filtering/search to list pages
4. Generate OG images dynamically
5. Enhance metadata (keywords, descriptions)
