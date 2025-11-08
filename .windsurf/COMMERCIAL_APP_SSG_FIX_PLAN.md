# Commercial App SSG Fix - Plan

**TEAM-XXX: CMS-friendly approach with reusable aside components**

## 🎯 Approach
1. ✅ **Created reusable aside components** (4 variants)
2. ✅ **Keep content in Props.tsx** (CMS-friendly)
3. ✅ **Render from config** in Page.tsx (no JSX serialization)

## Components Created

**Location:** `frontend/apps/commercial/components/organisms/HeroAsides/`

### 4 Aside Variants

1. **IconAside** - Icon + text (legal, simple callouts)
2. **ImageAside** - Images with captions (1536x1024, 1024x1024, 1024x1536)
3. **CardAside** - Cards with custom content (code examples)
4. **StatsAside** - Multiple stats with icons (metrics, social proof)

## The Pattern

### Props.tsx (CMS - All Content Here):

```tsx
import type { AsideConfig } from '../../organisms/HeroAsides'

export const heroProps: HeroTemplateProps = {
  title: "Your Title",
  subtitle: "Your subtitle",
  asideConfig: {
    variant: 'icon',
    icon: 'FileText',
    title: 'Legal Document',
    subtitle: 'Please read carefully'
  } as AsideConfig
}
```

### Page.tsx (Renderer - Just Displays):

```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...heroProps}
  aside={renderAside(heroProps.asideConfig)}
/>
```

## Benefits

✅ **Props.tsx = CMS** - All content stays in one place  
✅ **No JSX serialization** - Config objects are serializable  
✅ **Reusable components** - 4 variants cover all cases  
✅ **Type-safe** - AsideConfig union type  
✅ **Image support** - Automatic aspect ratio handling  

## Migration Steps

### Step 1: Update Props File

**Before (PageProps.tsx):**
```tsx
aside: (
  <div>
    <FileText />
    <p>Legal Document</p>
  </div>
)
```

**After (PageProps.ts):** ← Renamed to `.ts`
```typescript
asideConfig: {
  variant: 'icon',
  icon: 'FileText',
  title: 'Legal Document',
  subtitle: 'Please read carefully'
}
```

### Step 2: Rename Props File

```bash
# After removing all JSX
mv PageProps.tsx PageProps.ts
```

**Why?**
- Props files no longer contain JSX
- Only config objects and type imports
- Should use `.ts` not `.tsx`

### Step 3: Update Page.tsx

**Before:**
```tsx
import { FileText } from 'lucide-react'

<HeroTemplate 
  {...props}
  aside={<div><FileText /><p>Legal Document</p></div>}
/>
```

**After:**
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...props}
  aside={renderAside(props.asideConfig)}
/>
```

## Pages to Update

| # | Page | Current Aside | New Variant | Image Needed? |
|---|------|---------------|-------------|---------------|
| 1 | TermsPage | Icon (FileText) | IconAside | No |
| 2 | PrivacyPage | Icon (Shield) | IconAside | No |
| 3 | RhaiScriptingPage | ? | ImageAside | Yes - code viz |
| 4 | DevelopersPage | Card (code) | CardAside | No |
| 5 | ResearchPage | ? | ImageAside | Yes - academic |
| 6 | HomelabPage | ? | ImageAside | ✅ Has image |
| 7 | CommunityPage | ? | StatsAside | Optional |
| 8 | EducationPage | ? | ImageAside | Yes - learning |
| 9 | ProvidersPage | ? | ImageAside | ✅ Has image |
| 10 | StartupsPage | ? | ImageAside | Yes - growth |

## Image Generation

**Sizes needed:**
- 1536x1024 (landscape) - Wide content
- 1024x1024 (square) - Balanced
- 1024x1536 (portrait) - Tall content

**Pages needing images:**
- [ ] RhaiScriptingPage - Code/scripting (1024x1024)
- [ ] ResearchPage - Academic (1024x1536)
- [ ] CommunityPage - Optional stats or image
- [ ] EducationPage - Learning (1024x1536)
- [ ] StartupsPage - Growth (1536x1024)

**Existing images:**
- ✅ HomelabPage - `homelab-network.png`
- ✅ ProvidersPage - `gpu-earnings.png`
- ✅ Features pages - Various feature images

## Execution Plan

### Phase 1: Fix TermsPage (Example)
1. Update TermsPageProps.tsx with `asideConfig`
2. Update TermsPage.tsx to use `renderAside()`
3. Test build
4. Verify rendering

### Phase 2: Fix Remaining Pages
For each page:
1. Identify current aside content
2. Choose appropriate variant
3. Update Props.tsx with config
4. Update Page.tsx with renderAside()
5. Test build

### Phase 3: Generate Images
1. Create prompts for each page
2. Generate images (AI or design)
3. Optimize and save to `/public/images/`
4. Update Props.tsx with image paths

### Phase 4: Verify
1. Build commercial app
2. Build full monorepo
3. Test all pages render correctly
4. Verify images load properly

## Documentation

- **[HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md)** - Complete guide with examples
- **[COMMERCIAL_APP_SSG_FIX_PLAN.md](./COMMERCIAL_APP_SSG_FIX_PLAN.md)** - Original plan
- **[COMMERCIAL_APP_SSG_FIX_SUMMARY.md](./COMMERCIAL_APP_SSG_FIX_SUMMARY.md)** - Status summary

## Time Estimate

- Phase 1 (TermsPage example): 15 min
- Phase 2 (8 remaining pages): 1 hour
- Phase 3 (Image generation): 30-60 min (depends on AI/design)
- Phase 4 (Verification): 15 min

**Total: 2-2.5 hours**

## Rule Zero Compliance

✅ **Following Rule Zero:**
- Created reusable components (not one-off fixes)
- Updated architecture (proper separation of concerns)
- No backwards compatibility wrappers
- Deleted problematic patterns (JSX in props)

## Next Steps

1. **Review HERO_ASIDES_GUIDE.md** for detailed examples
2. **Start with TermsPage** as proof of concept
3. **Generate images** for pages that need them
4. **Update remaining pages** following the pattern
5. **Verify and test** everything builds

---

**Status:** Ready for execution  
**Approach:** CMS-friendly with reusable components  
**Components:** ✅ Created  
**Documentation:** ✅ Complete  
**Images:** 🔴 Need to generate 5 images
