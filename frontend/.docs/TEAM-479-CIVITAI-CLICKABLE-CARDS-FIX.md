# TEAM-479: CivitAI Clickable Cards Fix

**Date:** November 12, 2025  
**Status:** ✅ COMPLETE - Cards are now clickable

## Problem

CivitAI model cards on `/models/civitai` were NOT clickable. Unlike the HuggingFace page which used `<Link>` components directly, the CivitAI page used the `ModelCardVertical` component which had no href support.

## Root Cause

The `ModelCardVertical` component (used for CivitAI cards) did not have:
1. An `href` prop in its interface
2. A Link wrapper to make cards clickable
3. Any way to pass click handlers or navigation

## Solution

### 1. Updated `ModelCardVertical` Component

**File:** `/packages/rbee-ui/src/marketplace/organisms/ModelCardVertical/ModelCardVertical.tsx`

**Changes:**
- Added `href?: string` prop to `ModelCardVerticalProps` interface
- Imported Next.js `Link` component
- Wrapped card content in conditional Link wrapper
- Added `cursor-pointer` class for visual feedback

```tsx
export interface ModelCardVerticalProps {
  model: {
    // ... existing props
  }
  href?: string  // NEW: Optional href for clickable cards
}

export function ModelCardVertical({ model, href }: ModelCardVerticalProps) {
  // ... card rendering logic
  
  const cardContent = (
    <Card className="... cursor-pointer">
      {/* Card content */}
    </Card>
  )

  // TEAM-479: Wrap in Link if href is provided
  if (href) {
    return (
      <Link href={href} className="block h-full">
        {cardContent}
      </Link>
    )
  }

  return cardContent
}
```

### 2. Updated CivitAI Page

**File:** `/apps/marketplace/app/models/civitai/page.tsx`

**Changes:**
- Added `href` prop to each `ModelCardVertical` component
- Links to `/models/civitai/${model.id}`

```tsx
{models.map((model) => (
  <ModelCardVertical
    key={model.id}
    href={`/models/civitai/${model.id}`}  // NEW: Link to detail page
    model={{
      // ... model props
    }}
  />
))}
```

## Verification

### Puppeteer Test Results

✅ **Navigation Test:**
```
Before click: http://localhost:7823/models/civitai
After click:  http://localhost:7823/models/civitai/257749
```

✅ **Click Selector:** `a[href^="/models/civitai/"]` (successfully found and clicked)

✅ **Visual Feedback:** Cards have `cursor-pointer` class and hover effects

## Files Modified

1. `/packages/rbee-ui/src/marketplace/organisms/ModelCardVertical/ModelCardVertical.tsx`
   - Added href prop
   - Added Link import
   - Added conditional Link wrapper
   - Added cursor-pointer class

2. `/apps/marketplace/app/models/civitai/page.tsx`
   - Added href prop to ModelCardVertical components
   - Added TEAM-479 documentation comment

## Build Status

✅ TypeScript compilation successful
✅ No build errors
✅ UI package rebuilt successfully

## Known Issues

**Image Configuration Error (Separate Issue):**
The detail page shows an error about `image.civitai.com` not being configured in `next.config.js`. This is a Next.js image optimization issue, NOT a clickability issue.

**To Fix:**
Add to `next.config.js`:
```js
images: {
  remotePatterns: [
    {
      protocol: 'https',
      hostname: 'image.civitai.com',
    },
  ],
}
```

## Comparison: HuggingFace vs CivitAI

### HuggingFace (Already Working)
- Used `<Link>` directly in page component
- Simple card structure with inline content
- No reusable card component

### CivitAI (Now Fixed)
- Uses reusable `ModelCardVertical` component
- Complex card with image, badges, stats
- Now supports optional `href` prop for clickability

## Design Pattern

The solution follows a flexible design pattern:
- **With href:** Card is wrapped in Link (clickable)
- **Without href:** Card renders normally (non-clickable)

This allows the component to be used in both clickable and non-clickable contexts.

## Testing Checklist

- [x] Cards render correctly
- [x] Cards are clickable (Puppeteer verified)
- [x] Navigation works to detail pages
- [x] Hover effects work
- [x] Cursor pointer shows on hover
- [x] TypeScript types are correct
- [x] Build passes without errors

## Next Steps

1. Fix image configuration for CivitAI images (separate task)
2. Consider applying same pattern to other card components
3. Add similar href support to other marketplace card types

---

**Created by:** TEAM-479  
**Verified:** November 12, 2025  
**Status:** ✅ Cards are clickable and working
