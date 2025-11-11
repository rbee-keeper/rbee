# SSG Blocker: JSX in Props Files

**Date:** 2025-11-08  
**Status:** 🚨 BLOCKING SSG  
**Priority:** HIGH

---

## 🐛 Problem

All commercial site pages show as **Dynamic (ƒ)** instead of **Static (○)** because:

1. **Props files contain JSX** (React elements)
2. **JSX contains functions** (component references, event handlers)
3. **Functions cannot be serialized** for static generation
4. **`force-dynamic` was added** as a workaround but never removed

---

## 📊 Current State

### Build Output
```
Route (app)                                 Size  First Load JS    
├ ƒ /                                      144 B         561 kB  ❌ Dynamic
├ ƒ /pricing                               145 B         561 kB  ❌ Dynamic
├ ƒ /features                              145 B         561 kB  ❌ Dynamic
... (all 23 pages are dynamic)
```

### Root Cause
```tsx
// PricingPageProps.tsx
'use client'  // ❌ This makes it a Client Component

export const pricingHeroProps = {
  heading: (
    <>
      Free Forever.  // ❌ JSX = React.createElement() = function
      <br />
      <span className="text-primary">Premium Optional.</span>
    </>
  ),
  visual: (
    <div className="...">  // ❌ More JSX = more functions
      <picture>...</picture>
    </div>
  )
}
```

**Why it fails:**
1. Props file has `'use client'`
2. Props contain JSX (React elements)
3. React elements are objects with `$$typeof`, `type` (function), `props`
4. Next.js tries to serialize for SSG
5. **Error:** "Functions cannot be passed directly to Client Components"

---

## ✅ Solution Options

### Option 1: Convert JSX to Strings (Quick Fix)
**Pros:** Simple, works for text content  
**Cons:** Loses formatting, not suitable for complex layouts

```tsx
// Before
heading: (
  <>
    Free Forever.
    <br />
    <span className="text-primary">Premium Optional.</span>
  </>
)

// After
heading: "Free Forever. Premium Optional."
headingHighlight: "Premium Optional."
```

### Option 2: Render JSX in Page Component (Recommended)
**Pros:** Keeps JSX, proper separation, SSG-compatible  
**Cons:** Requires restructuring

```tsx
// Props file (NO 'use client', NO JSX)
export const pricingHeroProps = {
  badgeText: 'Lifetime Pricing',
  headingPart1: 'Free Forever.',
  headingPart2: 'Premium Optional.',
  description: '...',
}

// Page component ('use client', renders JSX)
'use client'
export function PricingPage() {
  return (
    <HeroTemplate
      {...pricingHeroProps}
      heading={
        <>
          {pricingHeroProps.headingPart1}
          <br />
          <span className="text-primary">{pricingHeroProps.headingPart2}</span>
        </>
      }
    />
  )
}
```

### Option 3: Use Markdown (Best for Content)
**Pros:** Clean, portable, SEO-friendly  
**Cons:** Requires markdown parser

```tsx
// Props file
export const pricingHeroProps = {
  heading: "Free Forever.\n\n**Premium Optional.**",
  description: "Core rbee is GPL-3.0..."
}

// Page component
import ReactMarkdown from 'react-markdown'

<ReactMarkdown>{pricingHeroProps.heading}</ReactMarkdown>
```

---

## 🎯 Recommended Approach

### Phase 1: Identify JSX Usage
```bash
# Find all JSX in props files
grep -r "heading: (" frontend/apps/commercial/components/pages/*/Props.tsx
grep -r "visual: (" frontend/apps/commercial/components/pages/*/Props.tsx
grep -r "<" frontend/apps/commercial/components/pages/*/Props.tsx
```

### Phase 2: Categorize by Complexity

**Simple Text (Option 1):**
- Headings with `<br />` and `<span>`
- Descriptions with basic formatting
- → Convert to strings with highlight markers

**Complex Layouts (Option 2):**
- Visual components with images
- Interactive elements
- → Move JSX to page component

**Content-Heavy (Option 3):**
- FAQ answers
- Long descriptions
- → Convert to markdown

### Phase 3: Implement Fixes

1. **Remove `'use client'` from all Props files**
   ```bash
   find frontend/apps/commercial/components/pages -name "*Props.tsx" -exec sed -i "/^'use client'$/d" {} \;
   ```

2. **Convert simple JSX to strings**
   ```tsx
   // Before
   heading: (<>Free Forever.<br /><span>Premium</span></>)
   
   // After
   heading: "Free Forever.",
   headingHighlight: "Premium Optional."
   ```

3. **Move complex JSX to page components**
   ```tsx
   // Props: Just data
   export const props = { title: "...", items: [...] }
   
   // Page: Renders JSX
   'use client'
   export function Page() {
     return <Template {...props} visual={<CustomVisual />} />
   }
   ```

4. **Remove `force-dynamic` declarations**
   ```bash
   find frontend/apps/commercial/app -name "page.tsx" -exec sed -i '/force-dynamic/d' {} \;
   ```

5. **Test build**
   ```bash
   pnpm --filter @rbee/commercial build
   # Should see ○ (Static) instead of ƒ (Dynamic)
   ```

---

## 📝 Files Affected

### Props Files with JSX (23 files)
```
components/pages/
├── HomePage/HomePageProps.tsx
├── PricingPage/PricingPageProps.tsx
├── FeaturesPage/FeaturesPageProps.tsx
├── ProvidersPage/ProvidersPageProps.tsx
├── UseCasesPage/UseCasesPageProps.tsx
├── HomelabPage/HomelabPageProps.tsx
├── EducationPage/EducationPageProps.tsx
├── StartupsPage/StartupsPageProps.tsx
├── EnterprisePage/EnterprisePageProps.tsx
├── DevelopersPage/DevelopersPageProps.tsx
├── DevOpsPage/DevOpsPageProps.tsx
├── ResearchPage/ResearchPageProps.tsx
├── SecurityPage/SecurityPageProps.tsx
├── CompliancePage/CompliancePageProps.tsx
├── CommunityPage/CommunityPageProps.tsx
├── LegalPage/LegalPageProps.tsx
├── PrivacyPage/PrivacyPageProps.tsx
├── TermsPage/TermsPageProps.tsx
├── HeterogeneousHardwarePage/HeterogeneousHardwarePageProps.tsx
├── MultiMachinePage/MultiMachinePageProps.tsx
├── OpenAICompatiblePage/OpenAICompatiblePageProps.tsx
├── RhaiScriptingPage/RhaiScriptingPageProps.tsx
└── ComparisonPage/ComparisonPageProps.tsx
```

### Page Files with force-dynamic (23 files)
```
app/
├── page.tsx
├── pricing/page.tsx
├── features/page.tsx
├── features/*/page.tsx (5 files)
├── earn/page.tsx
├── gpu-providers/page.tsx
├── use-cases/page.tsx
├── use-cases/*/page.tsx (2 files)
├── legal/page.tsx
├── legal/*/page.tsx (2 files)
├── compare/page.tsx
└── compare/*/page.tsx (4 files)
```

---

## 🚨 Why This Matters

### Current Impact
- **SEO:** Dynamic pages = slower indexing
- **Performance:** Server-side rendering on every request
- **Cost:** More server resources needed
- **UX:** Slower page loads (no pre-rendering)

### Expected Impact After Fix
- **SEO:** ✅ All pages pre-rendered, instant indexing
- **Performance:** ✅ Static files served from CDN
- **Cost:** ✅ Minimal server resources
- **UX:** ✅ Instant page loads

---

## 📊 Effort Estimate

| Task | Files | Effort | Priority |
|------|-------|--------|----------|
| Remove 'use client' from Props | 23 | 5 min | HIGH |
| Convert simple JSX to strings | ~15 | 2 hours | HIGH |
| Move complex JSX to pages | ~8 | 3 hours | MEDIUM |
| Test and verify | All | 1 hour | HIGH |
| **Total** | **23** | **~6 hours** | **HIGH** |

---

## ✅ Success Criteria

```bash
pnpm --filter @rbee/commercial build

# Should see:
Route (app)                                 Size  First Load JS    
├ ○ /                                      144 B         561 kB  ✅ Static
├ ○ /pricing                               145 B         561 kB  ✅ Static
├ ○ /features                              145 B         561 kB  ✅ Static
... (all pages static)

○  (Static)   prerendered as static content  ✅
ƒ  (Dynamic)  server-rendered on demand      ❌ (should be 0)
```

---

## 🔧 Quick Commands

```bash
# Check current state
pnpm --filter @rbee/commercial build | grep "Route (app)" -A 30

# Find JSX in props
grep -r "heading: (" frontend/apps/commercial/components/pages/

# Remove 'use client' from props
find frontend/apps/commercial/components/pages -name "*Props.tsx" -exec sed -i "/^'use client'$/d" {} \;

# Remove force-dynamic (after fixing JSX)
find frontend/apps/commercial/app -name "page.tsx" -exec sed -i '/force-dynamic/d' {} \;

# Test build
pnpm --filter @rbee/commercial build
```

---

## 📚 References

- [Next.js Static Generation](https://nextjs.org/docs/app/building-your-application/rendering/server-components#static-rendering-default)
- [Client Components](https://nextjs.org/docs/app/building-your-application/rendering/client-components)
- [Serialization Error](https://nextjs.org/docs/messages/prerender-error)

---

**Status:** 🚨 BLOCKING - All pages currently dynamic due to JSX in props  
**Next Step:** Implement Option 2 (move JSX to page components) for proper SSG support
