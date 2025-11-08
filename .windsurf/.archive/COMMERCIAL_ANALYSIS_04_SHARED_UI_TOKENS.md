# Shared UI & Theme Tokens

**Analysis Date:** 2025-11-06  
**Scope:** `frontend/packages/rbee-ui`  
**Purpose:** Shared component library and design system analysis

---

## Package Structure

**Location:** `frontend/packages/rbee-ui/src/`

**Organization:**
```
rbee-ui/src/
├── atoms/          (245 items) - Basic UI primitives
├── molecules/      (182 items) - Composite components
├── organisms/      (58 items) - Complex components
├── patterns/       (4 items) - Layout patterns
├── marketplace/    (58 items) - Marketplace-specific
├── tokens/         (4 items) - Design tokens
├── utils/          (5 items) - Utilities
├── hooks/          (2 items) - React hooks
├── icons/          (25 items) - Icon components
└── providers/      (5 items) - Context providers
```

**Total Components:** 500+ components across all levels

---

## Design Token System

### Theme Tokens File

**Location:** `src/tokens/theme-tokens.css`  
**Format:** CSS custom properties (HEX colors, NOT OKLCH)  
**Brand Color:** `#f59e0b` (amber/gold)

### Color Tokens

**Light Mode:**
```css
--background: #f3f4f6;      /* gray-100 - dimmed canvas */
--foreground: #0f172a;      /* slate-900 */
--card: #ffffff;            /* true white cards */
--primary: #d97706;         /* amber-600 - brand authority */
--accent: #f59e0b;          /* amber-500 - accents/hover */
--muted: #eef2f7;           /* table rows, subtle chips */
--border: #cbd5e1;          /* slate-300 */
```

**Dark Mode:**
```css
--background: #0f172a;      /* slate-900 */
--foreground: #f1f5f9;      /* slate-100 */
--card: #1e293b;            /* slate-800 */
--primary: #f59e0b;         /* amber-500 */
--accent: #fbbf24;          /* amber-400 */
--muted: #1e293b;           /* slate-800 */
--border: #334155;          /* slate-700 */
```

**Action Colors:**
```css
--success: #10b981;         /* emerald-500 */
--danger: #dc2626;          /* red-600 */
--destructive: #dc2626;     /* red-600 */
```

**Chart Colors:**
```css
--chart-1: #d97706;         /* amber-600 */
--chart-2: #3b82f6;         /* blue-500 */
--chart-3: #10b981;         /* emerald-500 */
--chart-4: #8b5cf6;         /* violet-500 */
--chart-5: #ef4444;          /* red-500 */
```

**Terminal Colors:**
```css
--terminal-red: #ef4444;
--terminal-amber: #d97706;
--terminal-green: #10b981;
--console-bg: #0f172a;
--console-fg: #f1f5f9;
```

**Syntax Highlighting:**
```css
--syntax-keyword: #2563eb;  /* blue-600 */
--syntax-import: #7c3aed;   /* violet-600 */
--syntax-string: #b45309;   /* amber-700 */
--syntax-function: #059669; /* emerald-600 */
--syntax-comment: #64748b;  /* slate-500 */
```

---

### Spacing Scale

```css
/* Spacing tokens */
--spacing-xs: 0.25rem;   /* 4px */
--spacing-sm: 0.5rem;    /* 8px */
--spacing-md: 1rem;      /* 16px */
--spacing-lg: 1.5rem;    /* 24px */
--spacing-xl: 2rem;      /* 32px */
--spacing-2xl: 3rem;     /* 48px */
--spacing-3xl: 4rem;     /* 64px */
```

**Common Usage:**
- Card padding: `p-6` (24px) or `p-8` (32px)
- Section gaps: `gap-6` (24px)
- Page margins: `mb-8` (32px), `mb-12` (48px)

---

### Typography Scale

```css
/* Font families */
--font-sans: 'Geist Sans', system-ui, sans-serif;
--font-serif: 'Source Serif 4', Georgia, serif;
--font-mono: 'Geist Mono', 'Courier New', monospace;
```

**Font Sizes:**
```css
--text-xs: 0.75rem;      /* 12px */
--text-sm: 0.875rem;     /* 14px */
--text-base: 1rem;       /* 16px */
--text-lg: 1.125rem;     /* 18px */
--text-xl: 1.25rem;      /* 20px */
--text-2xl: 1.5rem;      /* 24px */
--text-3xl: 1.875rem;    /* 30px */
--text-4xl: 2.25rem;     /* 36px */
--text-5xl: 3rem;        /* 48px */
```

---

### Border Radius

```css
--radius-sm: 0.25rem;    /* 4px */
--radius-md: 0.375rem;   /* 6px */
--radius-lg: 0.5rem;     /* 8px */
--radius-xl: 0.75rem;    /* 12px */
--radius-2xl: 1rem;      /* 16px */
--radius-full: 9999px;   /* Fully rounded */
```

---

### Elevation (Shadows)

```css
--shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
```

---

### Focus Ring System

```css
--focus-ring-color: var(--ring, #f59e0b);
--focus-ring-width: 2px;
--focus-ring-offset: 2px;
```

**Global consistency:** All interactive elements use same focus ring

---

## CSS Loading Strategy

### Globals File

**Location:** `src/tokens/globals.css`

```css
/* @import for fonts must come first */
@import "./fonts.css";

@import "tailwindcss";
@import "@repo/tailwind-config";
@import "./theme-tokens.css";

@source "../**/*.{ts,tsx}";
```

**Purpose:**
- ✅ Source file for Tailwind compilation
- ✅ Storybook imports this directly (enables JIT for arbitrary values)
- ✅ Apps import pre-built CSS: `import '@rbee/ui/styles.css'`
- ✅ `@source` directive scans UI package files for Tailwind classes
- ✅ Vite plugin (@tailwindcss/vite) handles JIT compilation

---

### App Integration Pattern

**Commercial Site (`apps/commercial/app/layout.tsx`):**
```typescript
import './globals.css'      // App CSS with JIT scanning
import '@rbee/ui/styles.css' // Pre-built tokens + components
```

**Marketplace (`apps/marketplace/app/layout.tsx`):**
```typescript
import "./globals.css";      // App CSS with JIT scanning
import "@rbee/ui/styles.css"; // Pre-built tokens + components
```

**Pattern:**
1. App-specific CSS first (enables arbitrary values in app)
2. UI library CSS second (provides tokens + component styles)
3. All fonts loaded in `@rbee/ui/styles.css`

---

## Marketplace-Specific Components

**Location:** `src/marketplace/`

### Structure (from README.md)

```
marketplace/
├── atoms/       (1 item)
├── molecules/   (6 items)
├── organisms/   (15 items)
├── templates/   (18 items)
├── pages/       (12 items)
├── hooks/       (3 items)
└── types/       (1 item)
```

**Total:** 56 marketplace-specific components

---

### Key Components

#### Organisms (15 components)
- **ModelCard** - Display single model with stats
- **WorkerCard** - Display worker binary
- **MarketplaceGrid** - Generic grid with loading/error states
- **FilterBar** - Search and sort controls

#### Templates (18 components)
- **ModelListTemplate** - Complete model browsing interface
- **ModelDetailTemplate** - Detailed model view
- **WorkerListTemplate** - Worker browsing interface
- **ArtifactDetailPageTemplate** - Generic detail page (TEAM-421)

#### Pages (12 components)
- **ModelsPage** - Complete models page (DUMB wrapper)
- **ModelDetailPage** - Complete model detail
- **WorkersPage** - Complete workers page (DUMB wrapper)

---

### Design Principles (from README.md)

1. **DUMB COMPONENTS** - No data fetching, only props
2. **REUSE ATOMS/MOLECULES** - Don't recreate Button, Card, Badge
3. **CONSISTENT STYLING** - Follow rbee-ui patterns
4. **SSG-READY** - All data in Props files
5. **TYPED** - Full TypeScript support
6. **FLEXIBLE** - Works in Next.js and Tauri

---

### Styling Consistency

**All marketplace components use:**
- Tailwind CSS classes
- Design tokens from `@rbee/ui/tokens`
- Consistent spacing: `p-6`, `gap-6`, etc.
- Responsive layouts (mobile-first)
- Dark mode support (via `dark:` prefix)

**Card Structure Pattern:**
```tsx
<Card className="p-6">
  <IconCardHeader /> {/* No manual spacing props */}
  <CardContent className="p-0">
    {/* Content */}
  </CardContent>
  <CardFooter>
    {/* Optional */}
  </CardFooter>
</Card>
```

**Note:** User has emphasized consistency - no mixed patterns allowed

---

## Commercial Site Components

**Location:** `apps/commercial/components/`

### Templates

**Location:** `apps/commercial/components/templates/`

**Available Templates:**
- BeforeAfterTemplate
- CardGridTemplate
- ComparisonTemplate
- CTATemplate
- EmailCapture
- FAQTemplate
- FeaturesTabs
- HeroTemplate
- HowItWorks
- PopularModelsTemplate
- PricingTemplate
- ProblemTemplate
- SolutionTemplate
- TechnicalTemplate
- UseCasesTemplate
- WhatIsRbee

**Total:** 16+ marketing templates

---

### Pages

**Location:** `apps/commercial/components/pages/`

**Available Pages (27 components):**
- CommunityPage
- ComparisonPage
- CompliancePage
- DevOpsPage
- DevelopersPage
- EducationPage
- EnterprisePage
- FeaturesPage
- HeterogeneousHardwarePage
- HomePage
- HomelabPage
- LegalPage
- MultiMachinePage
- OpenAICompatiblePage
- PricingPage
- PrivacyPage
- ProvidersPage
- RbeeVsOllamaPage
- RbeeVsRayKservePage
- RbeeVsTogetherAiPage
- RbeeVsVllmPage
- ResearchPage
- RhaiScriptingPage
- SecurityPage
- StartupsPage
- TermsPage
- UseCasesPage

**Pattern:** Each page imports templates and composes them

**Example (HomePage.tsx):**
```tsx
export default function HomePage() {
  return (
    <main className="bg-background">
      <HeroTemplate {...homeHeroProps} />
      <TemplateContainer {...whatIsRbeeContainerProps}>
        <WhatIsRbee {...whatIsRbeeProps} />
      </TemplateContainer>
      <TemplateContainer {...sixAdvantagesContainerProps}>
        <CardGridTemplate {...sixAdvantagesProps} />
      </TemplateContainer>
      {/* ... 10 more sections ... */}
    </main>
  )
}
```

---

## Branding & Metaphor Usage

### Bee Metaphor Components

**BeeArchitecture Component:**
- **Location:** `src/organisms/BeeArchitecture/BeeArchitecture.tsx`
- **Purpose:** Visualize queen/hive/worker architecture
- **Emojis Used:** 🐝 (bee), 👑 (queen), 🏠 (hive)

**Component Structure:**
```tsx
type WorkerNode = {
  id: string
  label: string
  kind: 'cuda' | 'metal' | 'cpu'
}

type BeeTopology =
  | { mode: 'single-pc'; hostLabel: string; workers: WorkerNode[] }
  | { mode: 'multi-host'; hosts: Array<{ hostLabel: string; workers: WorkerNode[] }> }
```

**Visual Representation:**
- Queen node: Primary color, special styling
- Hive nodes: Muted background
- Worker nodes: Color-coded by type (CUDA/Metal/CPU)

---

### Bee Metaphor in Copy

**From audit findings:**
- "Turn your GPUs into ONE unified colony" (HomePage)
- "unified swarm" (multiple pages)
- Bee emoji 🐝 in metadata descriptions
- "keeper / queen / hive / worker" terminology throughout

**Audit Warning:** "Bee metaphor overused (12+ instances on HomePage alone)"

**Current Usage:**
- ✅ Architecture diagrams (BeeArchitecture component)
- ✅ Technical terminology (queen-rbee, rbee-hive, workers)
- ⚠️ Marketing copy (potentially overused per audit)

---

## Shared Components Used by Both Sites

### From @rbee/ui/organisms

**Footer:**
- Used by: Commercial + Marketplace
- Shared component for consistency

**Navigation:**
- Commercial: Custom `Navigation` component
- Marketplace: Custom `MarketplaceNav` component
- **NOT shared** - different nav structures

---

### From @rbee/ui/molecules

**BrandLogo:**
- Used by: Both sites
- Consistent logo across all apps

**ThemeToggle:**
- Used by: Both sites
- Consistent dark mode toggle

**TemplateContainer:**
- Used by: Commercial site
- Wrapper for page sections

---

### From @rbee/ui/atoms

**Button, Card, Badge, IconButton:**
- Used by: Both sites
- Core UI primitives

**Pagination:**
- Available for marketplace
- Not currently used

---

## Theme Provider Integration

### Commercial Site

**Provider:** Custom `ThemeProvider` from `@/components/providers/ThemeProvider`  
**Configuration:**
```tsx
<ThemeProvider 
  attribute="class" 
  defaultTheme="system" 
  enableSystem 
  disableTransitionOnChange
>
```

---

### Marketplace Site

**Provider:** `ThemeProvider` from `next-themes` (direct import)  
**Configuration:**
```tsx
<ThemeProvider 
  attribute="class" 
  defaultTheme="system" 
  enableSystem 
  disableTransitionOnChange
>
```

**Note:** Same configuration, different import source (commercial has custom wrapper)

---

## Hexagonal/Hive Visual Patterns

**Current State:** No explicit hex pattern components found in grep

**Potential Locations:**
- Could be in CSS backgrounds
- Could be in SVG assets
- Could be in Storybook stories

**Audit Mentions:** "Bee metaphor overused" but doesn't specifically call out hex patterns

**Recommendation:** Search for:
- SVG hexagon patterns
- CSS clip-path hexagons
- Background images with hex grids

---

## Icon System

**Location:** `src/icons/` (25 icon components)

**Examples:**
- GitHubIcon
- (Others not enumerated)

**Integration:** Lucide React for most icons

**Usage:**
```tsx
import { GitHubIcon } from '@rbee/ui/icons'
import { BookOpen, ArrowRight } from 'lucide-react'
```

---

## Consistency Patterns

### Card Structure (User Requirement)

**Correct Pattern:**
```tsx
<Card className="p-6">
  <IconCardHeader 
    icon={<Icon />}
    title="Title"
    description="Description"
  />
  <CardContent className="p-0">
    {/* Content */}
  </CardContent>
</Card>
```

**Incorrect Patterns (BANNED):**
```tsx
// ❌ Manual h3/p instead of IconCardHeader
<Card className="p-6">
  <h3 className="mb-4">Title</h3>
  <p>Description</p>
</Card>

// ❌ Mixed spacing (mb-4, mb-6)
<Card className="p-6 mb-4">
  <IconCardHeader className="mb-6" />
</Card>
```

**User Emphasis:** "No mixed patterns" - consistency is HIGH PRIORITY

---

## Testing Infrastructure

**From marketplace README.md:**

```bash
pnpm test           # Unit tests (Vitest)
pnpm test:ct        # Component tests (Playwright)
pnpm storybook      # Visual testing
```

**Coverage:**
- Unit tests for logic
- Component tests for interactions
- Storybook stories for visual regression

---

## Summary

**Design System:**
- ✅ Comprehensive token system (colors, spacing, typography)
- ✅ HEX colors (NOT OKLCH)
- ✅ Brand color: Amber/gold (#f59e0b)
- ✅ Dark mode support
- ✅ Focus ring system
- ✅ Chart + terminal + syntax colors

**Component Library:**
- ✅ 500+ components (atoms, molecules, organisms)
- ✅ 56 marketplace-specific components
- ✅ 27 commercial page components
- ✅ 16+ marketing templates
- ✅ Shared Footer, BrandLogo, ThemeToggle

**Branding:**
- ✅ BeeArchitecture component (visual diagrams)
- ✅ Queen/Hive/Worker terminology
- ⚠️ Bee metaphor potentially overused in copy (per audit)
- ❓ Hex patterns (not found in grep, may exist in CSS/SVG)

**Consistency:**
- ✅ User requires strict consistency (no mixed patterns)
- ✅ Standardized card structure
- ✅ Consistent spacing (p-6, gap-6)
- ✅ Shared design tokens across both sites

**Integration:**
- ✅ Both sites import `@rbee/ui/styles.css`
- ✅ App-specific CSS first, then UI library CSS
- ✅ Tailwind JIT compilation
- ✅ Theme providers configured identically

**Gaps:**
- No explicit hex/hive pattern components found
- Commercial and marketplace have separate nav components (not shared)
- Some duplication between commercial templates and marketplace templates
