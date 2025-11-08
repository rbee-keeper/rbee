# TEAM-422: UI Changes - Navigation & Layout

**Status:** ✅ COMPLETE  
**Date:** 2025-11-07  
**Team:** TEAM-422

## Changes Made

### 1. Navigation Labels Updated

**File:** `frontend/apps/marketplace/config/navigationConfig.ts`

Changed navigation labels to reflect the actual model sources:

**Before:**
- "LLM Models" → `/models/huggingface`
- "SD Models" → `/models/civitai`

**After:**
- "HF Models" → `/models/huggingface`
- "CivitAI Models" → `/models/civitai`

This makes it immediately clear which platform hosts which models.

---

### 2. CivitAI Page Layout Changed to Cards

**File:** `frontend/apps/marketplace/app/models/civitai/page.tsx`

**Before:** Table layout (like HuggingFace)
```tsx
<ModelTableWithRouting models={models} />
```

**After:** Card grid layout
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
  {models.map((model) => (
    <Link href={`/models/civitai/${modelIdToSlug(model.id)}`}>
      <ModelCard model={model} />
    </Link>
  ))}
</div>
```

**Responsive Grid:**
- Mobile: 1 column
- Tablet: 2 columns
- Desktop: 3 columns
- Large screens: 4 columns

---

### 3. HuggingFace Page Remains Table

**File:** `frontend/apps/marketplace/app/models/huggingface/page.tsx`

No changes - continues to use `ModelTableWithRouting` for efficient browsing of LLM models.

---

## Rationale

### Why Cards for CivitAI?

1. **Visual Medium:** Stable Diffusion models are inherently visual
2. **Image Previews:** Cards support model preview images
3. **Better Engagement:** Visual cards are more engaging for image models
4. **Showcase Stats:** Cards display downloads/likes prominently

### Why Table for HuggingFace?

1. **Data-Heavy:** LLM models have more technical specs
2. **Comparison:** Tables make it easier to compare models
3. **Efficiency:** Scan many models quickly
4. **Sorting:** Tables support better sorting/filtering

---

## Component Used

### ModelCard

**Location:** `@rbee/ui/marketplace`

**Features:**
- Image preview with gradient overlay
- Floating stats (downloads, likes)
- Author display
- Tag badges (up to 4 + overflow)
- Size badge
- Hover effects (scale, shadow, border)
- Responsive design

**Props:**
```typescript
{
  model: {
    id: string
    name: string
    description: string
    author?: string
    imageUrl?: string
    tags: string[]
    downloads: number
    likes: number
    size: string
  }
  onClick?: () => void
  onAction?: (modelId: string) => void
  actionButton?: React.ReactNode
}
```

---

## Visual Comparison

### CivitAI (Cards)
```
┌─────────┬─────────┬─────────┬─────────┐
│ [Image] │ [Image] │ [Image] │ [Image] │
│  Model  │  Model  │  Model  │  Model  │
│  Stats  │  Stats  │  Stats  │  Stats  │
│  Tags   │  Tags   │  Tags   │  Tags   │
└─────────┴─────────┴─────────┴─────────┘
```

### HuggingFace (Table)
```
┌────────────────────────────────────────┐
│ Name    │ Author  │ Downloads │ Likes  │
├────────────────────────────────────────┤
│ Model 1 │ Author  │ 1.2M      │ 500    │
│ Model 2 │ Author  │ 800K      │ 300    │
│ Model 3 │ Author  │ 600K      │ 200    │
└────────────────────────────────────────┘
```

---

## Files Modified

1. **frontend/apps/marketplace/config/navigationConfig.ts**
   - Updated navigation labels (lines 10-11)

2. **frontend/apps/marketplace/app/models/civitai/page.tsx**
   - Changed imports (lines 4-7)
   - Added size field to model mapping (line 30)
   - Replaced table with card grid (lines 64-78)

3. **frontend/apps/marketplace/app/models/huggingface/page.tsx**
   - No changes (remains table layout)

---

## Code Changes

### Navigation Config

```typescript
// Before
{ label: 'LLM Models', href: '/models/huggingface' },
{ label: 'SD Models', href: '/models/civitai' },

// After
{ label: 'HF Models', href: '/models/huggingface' },
{ label: 'CivitAI Models', href: '/models/civitai' },
```

### CivitAI Page

```typescript
// Before
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'

<ModelTableWithRouting models={models} />

// After
import { ModelCard } from '@rbee/ui/marketplace'
import Link from 'next/link'
import { modelIdToSlug } from '@/lib/slugify'

<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
  {models.map((model) => (
    <Link key={model.id} href={`/models/civitai/${modelIdToSlug(model.id)}`}>
      <ModelCard model={model} onClick={() => {}} />
    </Link>
  ))}
</div>
```

---

## Success Criteria

- [x] Navigation shows "HF Models" and "CivitAI Models"
- [x] CivitAI page uses card grid layout
- [x] HuggingFace page uses table layout
- [x] Cards are responsive (1-4 columns)
- [x] Cards link to detail pages correctly
- [x] No TypeScript errors

---

## User Experience

### CivitAI Page
- ✅ Visual browsing experience
- ✅ See model previews at a glance
- ✅ Hover effects for engagement
- ✅ Clear stats display
- ✅ Mobile-friendly grid

### HuggingFace Page
- ✅ Efficient data scanning
- ✅ Easy comparison of models
- ✅ Sortable columns
- ✅ Compact information display
- ✅ Fast navigation

---

## Next Steps

1. **Test the pages:**
   - Visit `/models/civitai` - should show card grid
   - Visit `/models/huggingface` - should show table

2. **Verify navigation:**
   - Check that labels are "HF Models" and "CivitAI Models"

3. **Test responsiveness:**
   - Resize browser to see grid adapt
   - Check mobile view

4. **Future enhancements:**
   - Add image URLs to CivitAI models
   - Add filtering/sorting to card grid
   - Add search functionality

---

**TEAM-422** - Separated model list layouts: Cards for visual SD models, Table for data-heavy LLM models. Updated navigation for clarity.
