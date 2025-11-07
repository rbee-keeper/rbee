# Step 4: Update Apps to Use Configs

**TEAM-460** | Phase 4 of 5

## Objective

Update commercial and marketplace apps to use the Navigation component with their respective configs.

## Commercial App Update

**File:** `frontend/apps/commercial/components/CommercialNav.tsx`

**Before:**
```tsx
import { Navigation } from '@rbee/ui/organisms/Navigation'

export function CommercialNav() {
  return <Navigation />  // Uses hardcoded values
}
```

**After:**
```tsx
import { Navigation } from '@rbee/ui/organisms/Navigation'
import { commercialNavConfig } from '@/config/navigationConfig'

export function CommercialNav() {
  return <Navigation config={commercialNavConfig} />
}
```

**Changes:**
1. Import config
2. Pass config to Navigation
3. Delete any custom navigation code

## Marketplace App Update

**File:** `frontend/apps/marketplace/components/MarketplaceNav.tsx`

**Before (current - 81 lines):**
```tsx
'use client'

import { SimpleNavigation } from '@rbee/ui/organisms/Navigation/SimpleNavigation'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
// ... 70+ lines of custom navigation
```

**After:**
```tsx
'use client'

import { Navigation } from '@rbee/ui/organisms/Navigation'
import { marketplaceNavConfig } from '@/config/navigationConfig'

export function MarketplaceNav() {
  return <Navigation config={marketplaceNavConfig} />
}
```

**Changes:**
1. Delete ALL custom navigation code
2. Import Navigation and config
3. Pass config to Navigation
4. **81 lines → 7 lines**

## Layout Files

### Commercial Layout

**File:** `frontend/apps/commercial/app/layout.tsx`

**No changes needed** - Already uses `<CommercialNav />`

### Marketplace Layout

**File:** `frontend/apps/marketplace/app/layout.tsx`

**No changes needed** - Already uses `<MarketplaceNav />`

## Verification Steps

### 1. Commercial App
```bash
cd frontend/apps/commercial
pnpm dev
```

**Check:**
- [ ] Platform dropdown works
- [ ] Solutions dropdown works
- [ ] Industries dropdown works (2-column layout)
- [ ] Resources dropdown works
- [ ] Docs link works
- [ ] GitHub link works
- [ ] "Join Waitlist" CTA works
- [ ] Mobile menu works
- [ ] Theme toggle works
- [ ] All analytics events fire

### 2. Marketplace App
```bash
cd frontend/apps/marketplace
pnpm dev
```

**Check:**
- [ ] "LLM Models" link works
- [ ] "SD Models" link works
- [ ] "More models" shows "Soon" badge and is disabled
- [ ] Separator shows between groups
- [ ] "LLM Workers" link works
- [ ] "Image Workers" link works
- [ ] "More workers" shows "Soon" badge and is disabled
- [ ] Docs link works
- [ ] GitHub link works
- [ ] "Back to rbee.dev" CTA works
- [ ] Theme toggle works

## Rules

- ✅ **DELETE** hand-rolled navigation code
- ✅ **USE** Navigation component with config
- ✅ **VERIFY** both apps work
- ❌ **DO NOT** keep duplicate code
- ❌ **DO NOT** skip verification

## Files Modified

```
frontend/apps/commercial/
├── config/
│   └── navigationConfig.ts (created in Step 3)
└── components/
    └── CommercialNav.tsx (updated)

frontend/apps/marketplace/
├── config/
│   └── navigationConfig.ts (created in Step 3)
└── components/
    └── MarketplaceNav.tsx (updated - 81 lines → 7 lines)
```

---

**Next:** `05_STEP_5_CLEANUP.md`
