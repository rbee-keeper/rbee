# Commercial Site: SEO & Metadata Overview

**Analysis Date:** 2025-11-06  
**Scope:** `frontend/apps/commercial`  
**Purpose:** Complete SEO metadata inventory

---

## SEO Infrastructure

### Centralized Utilities

**File:** `lib/seo/structured-data.ts`  
**Purpose:** JSON-LD schema generation  
**Created by:** TEAM-454 (Aggressive SEO initiative)

**Available Schemas:**
1. `getOrganizationSchema()` - Organization info
2. `getWebSiteSchema()` - Website + SearchAction
3. `getPremiumProductSchemas()` - 4 pricing products
4. `getFeatureFAQSchema(feature)` - Feature-specific FAQs
5. `generateJSONLD(schemas)` - Wrapper for multiple schemas

**Schema Details:**
- Organization: rbee company info, social links, contact
- WebSite: Search action targeting `/search?q={term}`
- Products: Premium Queen (€129), GDPR (€249), Bundle (€279), Complete (€499)
- All products: PreOrder availability, valid until 2026-06-30
- Products have aggregateRating (5 stars, 7-15 reviews) - **LIKELY FABRICATED**

---

## Per-Route Metadata Analysis

### ✅ `/` (Home) - COMPLETE

**File:** `app/page.tsx`

**Metadata Export:**
```typescript
export const metadata: Metadata = {
  alternates: { canonical: '/' },
  description: '🐝 Turn your GPUs into ONE unified colony...',
  keywords: [12 terms including 'rbee vs Ollama', 'rbee vs vLLM'],
}
```

**JSON-LD Schemas:**
- ✅ Organization schema
- ✅ WebSite schema with SearchAction

**Title:** Inherited from layout.tsx  
**Description:** Aggressive CTR-focused with emoji  
**Keywords:** 12 commercial + competitive terms  
**Canonical:** `/`  
**OG/Twitter:** Inherited from layout  

**Issues:**
- Uses emoji in description (🐝) - may not render in all contexts
- "unified colony" bee metaphor (audit flags as overused)

---

### ✅ `/pricing` - COMPLETE

**File:** `app/pricing/page.tsx`

**Metadata Export:**
```typescript
export const metadata: Metadata = {
  title: 'rbee Pricing - Free Forever or €129-499 Lifetime | No Subscriptions',
  description: '🐝 Free forever (GPL-3.0) or €129-499 lifetime...',
  keywords: [11 terms],
  alternates: { canonical: '/pricing' },
  openGraph: { ... },
  twitter: { ... },
}
```

**JSON-LD Schemas:**
- ✅ 4 Product schemas (Premium Queen, GDPR, Bundle, Complete)

**Title:** Full, descriptive, includes pricing range  
**Description:** Emphasizes lifetime pricing, no subscriptions  
**Keywords:** 11 pricing-focused terms  
**Canonical:** `/pricing`  
**OG:** Custom title, description, type: website  
**Twitter:** Custom card (summary_large_image)  

**Issues:**
- Product schemas have aggregateRating with no real reviews (pre-launch)
- "99.3% cheaper" claim in description (needs sourcing)

---

### ❌ `/features` - MISSING

**File:** `app/features/page.tsx`

**Metadata Export:** NOT FOUND  
**JSON-LD Schemas:** NONE  

**Current State:**
- No metadata export
- Inherits layout.tsx defaults
- Title will be: "rbee - Stop Paying for AI APIs..."
- Description will be generic

**Needed:**
- Custom title: "rbee Features - 6 Unique Advantages..."
- Description highlighting 6 features
- Keywords: feature-specific terms
- Canonical URL
- Possibly FAQ schema for features

---

### ❌ `/use-cases` - MISSING

**File:** `app/use-cases/page.tsx`

**Metadata Export:** NOT FOUND  
**JSON-LD Schemas:** NONE  

**Needed:**
- Custom title
- Description
- Keywords
- Canonical URL

---

### ❌ `/compare/rbee-vs-ollama` - MISSING

**File:** `app/compare/rbee-vs-ollama/page.tsx`

**Metadata Export:** NOT FOUND  
**JSON-LD Schemas:** NONE  

**Needed (HIGH PRIORITY - audit rates 9/10 SEO value):**
- Title: "rbee vs Ollama - Feature Comparison 2025"
- Description: Key differentiators
- Keywords: "rbee vs ollama", "ollama alternative"
- Canonical URL
- Possibly Comparison schema

**Note:** Audit calls this page "excellent" - should have full SEO treatment

---

### ❌ `/compare/rbee-vs-vllm` - MISSING
### ❌ `/compare/rbee-vs-together-ai` - MISSING
### ❌ `/compare/rbee-vs-ray-kserve` - MISSING

**All comparison pages:** No metadata exports found

---

### ❌ `/features/*` Subroutes (6 pages) - MISSING

**All feature subroutes:** No metadata exports found

**Pages:**
- `/features/multi-machine`
- `/features/heterogeneous-hardware`
- `/features/ssh-deployment`
- `/features/rhai-scripting`
- `/features/openai-compatible`
- `/features/gdpr-compliance`

**Needed:**
- Custom titles per feature
- Feature-specific descriptions
- Keywords per feature
- Canonical URLs
- FAQ schemas (available in structured-data.ts but not used)

---

### ❌ `/industries/*` Routes (10 pages) - MISSING

**All industry pages:** No metadata exports found

**Critical Pages:**
- `/industries/education` - Audit rates 3/10 SEO value (worst)
- `/industries/research` - Audit rates 6/10
- `/industries/homelab` - Audit rates 4/10

**Needed:**
- Custom titles per industry
- Industry-specific descriptions
- Keywords per industry
- Canonical URLs

---

### ❌ `/community` - MISSING
### ❌ `/developers` - MISSING
### ❌ `/enterprise` - MISSING
### ❌ `/gpu-providers` - MISSING
### ❌ `/security` - MISSING
### ❌ `/legal` - MISSING

**All other top-level pages:** No metadata exports found

---

## Layout-Level Metadata

**File:** `app/layout.tsx`

**Global Metadata:**
```typescript
export const metadata: Metadata = {
  title: 'rbee - Stop Paying for AI APIs. Run Everything Free.',
  description: 'Turn your GPUs into an intelligent swarm...',
  keywords: [12 terms],
  authors: [{ name: 'rbee' }],
  creator: 'rbee',
  publisher: 'rbee',
  metadataBase: new URL('https://rbee.dev'),
  alternates: { canonical: '/' },
  openGraph: { ... },
  twitter: { ... },
  robots: { ... },
}
```

**Applies to:** All pages without custom metadata  
**Issues:**
- Generic title/description not suitable for all pages
- Canonical always points to `/` (wrong for subpages)

---

## Metadata Coverage Summary

| Route | Title | Description | Keywords | Canonical | OG/Twitter | JSON-LD | Status |
|-------|-------|-------------|----------|-----------|------------|---------|--------|
| `/` | ✅ (layout) | ✅ | ✅ | ✅ | ✅ (layout) | ✅ (2 schemas) | COMPLETE |
| `/pricing` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (4 schemas) | COMPLETE |
| `/features` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |
| `/use-cases` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |
| `/compare/*` (4) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |
| `/features/*` (6) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |
| `/industries/*` (10) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |
| Other (6) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | MISSING |

**Coverage:** 2/32 pages (6.25%)  
**Missing:** 30 pages need metadata

---

## JSON-LD Schema Usage

### Currently Used

**Home (`/`):**
- Organization schema
- WebSite schema with SearchAction

**Pricing (`/pricing`):**
- 4 Product schemas (Premium Queen, GDPR, Bundle, Complete)

### Available But Unused

**Feature FAQ schemas:**
- `getFeatureFAQSchema('openai-compatible')` - exists but not used
- Other features: No FAQ schemas defined

**Missing Schema Types:**
- BreadcrumbList (for navigation)
- Article (for blog/content pages)
- HowTo (for tutorials)
- Comparison (for vs pages)

---

## Inconsistencies & Issues

### 1. Duplicate/Conflicting Schemas

**Issue:** No conflicts found (only 2 pages have schemas)  
**Risk:** When adding schemas to other pages, need to ensure:
- Organization schema only on home (or use @id references)
- Product schemas only on pricing
- No duplicate WebSite schemas

### 2. Canonical URL Issues

**Issue:** Layout sets canonical to `/` for all pages  
**Impact:** Subpages inherit wrong canonical  
**Fix Needed:** Each page must override with correct canonical

### 3. Title/Description Duplication

**Issue:** 30 pages use same generic title/description from layout  
**Impact:** Poor SEO, low CTR, keyword cannibalization  
**Fix Needed:** Custom metadata for each page

### 4. Fabricated Data in Schemas

**Issue:** Product schemas have aggregateRating (5 stars, 7-15 reviews)  
**Reality:** Pre-launch product, no real reviews  
**Risk:** Schema.org violation, Google penalty  
**Fix Needed:** Remove aggregateRating or mark as "0 reviews"

### 5. Missing Keywords

**Issue:** Only 2 pages have keywords defined  
**Impact:** Lost opportunity for keyword targeting  
**Fix Needed:** Research and add keywords per page

### 6. No Breadcrumbs

**Issue:** No BreadcrumbList schema anywhere  
**Impact:** Lost rich snippet opportunity  
**Fix Needed:** Add breadcrumbs to all subpages

---

## OG Image Strategy

**Current State:** No OG images defined (except layout default)  
**Needed:**
- Dynamic OG image generation per page
- API route: `/api/og/[...slug]` (Next.js pattern)
- Images for: pricing, features, comparisons, industries

**Example Implementation:**
```typescript
// app/pricing/page.tsx
openGraph: {
  images: ['/api/og/pricing'],
}
```

---

## Recommendations by Priority

### 🔴 CRITICAL (Before Launch)

1. **Remove fabricated aggregateRating** from Product schemas
   - Current: 5 stars, 7-15 reviews (fake)
   - Fix: Remove or set reviewCount: "0"

2. **Fix canonical URLs** on all pages
   - Current: All pages inherit canonical: `/`
   - Fix: Each page sets correct canonical

3. **Add metadata to comparison pages**
   - `/compare/rbee-vs-ollama` rated 9/10 SEO value
   - Needs full metadata treatment

### ⚠️ HIGH PRIORITY (Pre-Launch)

4. **Add metadata to all 30 missing pages**
   - Custom title, description, keywords
   - Canonical URLs
   - OG/Twitter cards

5. **Add BreadcrumbList schema** to all subpages
   - Improves navigation
   - Rich snippets in search

6. **Implement OG image generation**
   - Dynamic images per page
   - Better social sharing

### ✅ MEDIUM PRIORITY (Post-Launch)

7. **Add FAQ schemas** to feature pages
   - Already defined in structured-data.ts
   - Just need to wire up

8. **Add Article schemas** to content pages
   - For blog posts, guides, tutorials

9. **Optimize keywords** per page
   - Research search volume
   - Assign primary keywords
   - Avoid cannibalization

---

## Metadata Template (for missing pages)

```typescript
// Example: app/features/page.tsx
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Features - 6 Unique Advantages | Multi-Machine GPU Orchestration',
  description: 'Discover rbee\'s 6 unique advantages: Multi-machine, Heterogeneous hardware, SSH deployment, Rhai scripting, OpenAI API, GDPR compliance. Free forever.',
  keywords: [
    'rbee features',
    'multi-machine orchestration',
    'heterogeneous GPU',
    'SSH deployment',
    'Rhai scripting',
    'OpenAI compatible',
    'GDPR compliance',
  ],
  alternates: {
    canonical: '/features',
  },
  openGraph: {
    title: 'rbee Features - 6 Unique Advantages',
    description: 'Multi-machine, heterogeneous hardware, SSH, Rhai, OpenAI API, GDPR. Free forever.',
    type: 'website',
    url: 'https://rbee.dev/features',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Features - 6 Unique Advantages',
    description: 'Multi-machine, heterogeneous hardware, SSH, Rhai, OpenAI API, GDPR.',
  },
}
```

---

## Summary

**Current State:**
- ✅ 2 pages have complete metadata (Home, Pricing)
- ✅ Centralized schema utilities exist
- ❌ 30 pages missing metadata (93.75%)
- ❌ Fabricated review data in schemas
- ❌ No OG images
- ❌ No breadcrumbs

**Biggest Gaps:**
1. Comparison pages (high SEO value, no metadata)
2. Feature pages (6 pages, no metadata)
3. Industry pages (10 pages, no metadata)
4. Fabricated aggregateRating data

**Estimated Work:**
- Remove fake ratings: 1 hour
- Add metadata to 30 pages: 2-3 days
- Implement OG images: 1-2 days
- Add breadcrumbs: 1 day

**Total:** ~1 week of focused SEO work
