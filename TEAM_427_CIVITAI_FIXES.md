# TEAM-427: CivitAI Model Page Fixes

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE

## Issues Fixed

### 1. ✅ Wrong Links in InstallCTA

**Problem:**
- "Download rbee" linked to `/download` (404)
- "Learn More" linked to `/docs/getting-started` (wrong)
- "View on HuggingFace" shown on CivitAI models (wrong provider)

**Solution:**
- "Download rbee" → `https://docs.rbee.dev/docs/getting-started/installation`
- "Learn More" → `https://rbee.dev/`
- "View on CivitAI" → `https://civitai.com/models/{id}` (for CivitAI models)

### 2. ✅ Missing Image Gallery

**Problem:**
CivitAI models have multiple example images but they weren't being displayed.

**Solution:**
Added image gallery section showing up to 5 non-NSFW images in a responsive grid.

## Changes Made

### Files Modified

#### 1. `/frontend/apps/marketplace/components/InstallCTA.tsx`
Fixed CTA button links:

```tsx
// OLD
<a href="/download">Download rbee</a>
<a href="/docs/getting-started">Learn More</a>

// NEW
<a href="https://docs.rbee.dev/docs/getting-started/installation">Download rbee</a>
<a href="https://rbee.dev">Learn More</a>
```

#### 2. `/frontend/packages/rbee-ui/src/marketplace/templates/ModelDetailPageTemplate/ModelDetailPageTemplate.tsx`

**Added to interface:**
```typescript
export interface ModelDetailData {
  // ... existing fields
  
  // TEAM-427: CivitAI specific (optional)
  images?: Array<{ url: string; nsfw?: boolean }>
  externalUrl?: string // CivitAI or HuggingFace URL
  externalLabel?: string // "View on CivitAI" or "View on HuggingFace"
}
```

**Added image gallery:**
```tsx
{model.images && model.images.length > 0 && (
  <section>
    <Card>
      <CardHeader>
        <CardTitle>Example Images</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {model.images.map((image, index) => (
            <div key={index} className="relative aspect-square rounded-lg overflow-hidden bg-muted">
              <img
                src={image.url}
                alt={`Example ${index + 1}`}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  </section>
)}
```

**Updated external link:**
```tsx
// OLD
const hfUrl = huggingFaceUrl || `https://huggingface.co/${model.id}`
secondaryAction={{
  label: 'View on HuggingFace',
  href: hfUrl,
}}

// NEW
const externalUrl = model.externalUrl || huggingFaceUrl || `https://huggingface.co/${model.id}`
const externalLabel = model.externalLabel || 'View on HuggingFace'
secondaryAction={{
  label: externalLabel,
  href: externalUrl,
}}
```

#### 3. `/frontend/apps/marketplace/app/models/civitai/[slug]/page.tsx`

Added CivitAI-specific data:

```typescript
const model = {
  // ... existing fields
  images: latestVersion?.images?.filter((img) => !img.nsfw).slice(0, 5) || [],
  // TEAM-427: External link to CivitAI
  externalUrl: `https://civitai.com/models/${civitaiModel.id}`,
  externalLabel: 'View on CivitAI',
}
```

## Verification

### ✅ CivitAI Model Page
**URL:** https://main.rbee-marketplace.pages.dev/models/civitai/civitai-4201

**Verified:**
- ✅ "Download rbee" links to docs.rbee.dev installation page
- ✅ "Learn More" links to rbee.dev homepage
- ✅ "View on CivitAI" button (not HuggingFace)
- ✅ "View on CivitAI" links to https://civitai.com/models/4201
- ✅ Image gallery displays 5 example images
- ✅ Images in responsive 2-3 column grid
- ✅ Hover effect on images (scale-105)

### ✅ HuggingFace Model Page
**URL:** https://main.rbee-marketplace.pages.dev/models/huggingface/sentence-transformers--all-minilm-l6-v2

**Verified:**
- ✅ "Download rbee" links to docs.rbee.dev installation page
- ✅ "Learn More" links to rbee.dev homepage
- ✅ "View on HuggingFace" button (correct for HF models)
- ✅ No image gallery (HF models don't have images)

## Image Gallery Features

- **Responsive grid:** 2 columns on mobile, 3 columns on desktop
- **Aspect ratio:** Square (1:1) for consistent layout
- **Image count:** Up to 5 images (filtered for non-NSFW)
- **Hover effect:** Slight zoom (scale-105) on hover
- **Loading state:** Gray background while images load
- **Object fit:** Cover (fills the square without distortion)

## URL Structure

### CivitAI Models
- **Marketplace:** `/models/civitai/civitai-{id}`
- **External link:** `https://civitai.com/models/{id}`
- **Button label:** "View on CivitAI"

### HuggingFace Models
- **Marketplace:** `/models/huggingface/{slug}`
- **External link:** `https://huggingface.co/{id}`
- **Button label:** "View on HuggingFace"

## Next Steps

**None required.** All issues fixed and deployed.

---

**TEAM-427 SIGNATURE:** CivitAI model page fixed with correct links and image gallery.
