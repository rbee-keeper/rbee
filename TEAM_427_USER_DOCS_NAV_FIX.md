# TEAM-427: User-Docs Navigation Fixes

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE  
**Deployment URL:** https://main.rbee-user-docs.pages.dev

## Issues Fixed

### 1. ✅ Content Overlap with Fixed Navigation

**Problem:**
Page content was being hidden behind the fixed navigation bar at the top of the page.

**Root Cause:**
The Navigation component uses `fixed top-0` positioning, but the `#main` element had no padding-top to account for the navigation height.

**Solution:**
Added padding-top to `#main` in `globals.css`:

```css
@layer base {
    #main {
        padding-top: 4rem; /* h-16 mobile */
    }

    @media (min-width: 768px) {
        #main {
            padding-top: 3.5rem; /* h-14 desktop */
        }
    }
}
```

**Why these values:**
- Navigation height is `h-16` (4rem) on mobile
- Navigation height is `h-14` (3.5rem) on desktop (md breakpoint)
- Padding matches navigation height exactly

### 2. ✅ Missing Navigation Links

**Problem:**
Navigation didn't link to the homepage (rbee.dev) or marketplace (marketplace.rbee.dev).

**Solution:**
Updated `config/navigationConfig.ts`:

**Before:**
```typescript
{
  logoHref: '/docs',
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'Docs', href: '/docs' },
        { label: 'Quick Start', href: '/docs/getting-started/installation' },
        { label: 'API', href: '/docs/reference/api-openai-compatible' },
        { label: 'Architecture', href: '/docs/architecture/overview' },
      ],
    },
  ],
  actions: {
    cta: {
      label: 'rbee.dev',
      href: 'https://rbee.dev',
    },
  },
}
```

**After:**
```typescript
{
  logoHref: 'https://rbee.dev',
  sections: [
    {
      type: 'linkGroup',
      links: [
        { label: 'Home', href: 'https://rbee.dev' },
        { label: 'Marketplace', href: 'https://marketplace.rbee.dev' },
        { label: 'Docs', href: '/docs' },
        { label: 'Quick Start', href: '/docs/getting-started/installation' },
        { label: 'API', href: '/docs/reference/api-openai-compatible' },
        { label: 'Architecture', href: '/docs/architecture/overview' },
      ],
    },
  ],
  actions: {
    cta: {
      label: 'Download',
      href: '/docs/getting-started/installation',
      ariaLabel: 'Download rbee',
    },
  },
}
```

**Changes:**
- ✅ Logo now links to rbee.dev homepage
- ✅ Added "Home" link to rbee.dev
- ✅ Added "Marketplace" link to marketplace.rbee.dev
- ✅ Changed CTA from "rbee.dev" to "Download" (installation page)
- ✅ Reordered links for better UX flow

## Files Modified

### 1. `frontend/apps/user-docs/app/globals.css`
Added padding-top to prevent content overlap:
```css
#main {
    padding-top: 4rem; /* mobile */
}

@media (min-width: 768px) {
    #main {
        padding-top: 3.5rem; /* desktop */
    }
}
```

### 2. `frontend/apps/user-docs/config/navigationConfig.ts`
Updated navigation links and CTA:
- Logo href: `/docs` → `https://rbee.dev`
- Added "Home" and "Marketplace" links
- CTA: "rbee.dev" → "Download"

## Navigation Structure

### Desktop Navigation
```
[Logo] Home | Marketplace | Docs | Quick Start | API | Architecture [GitHub] [Download]
```

### Mobile Navigation
```
[Logo]                                                              [Menu]
```

When menu is open:
```
Home
Marketplace
Docs
Quick Start
API
Architecture
```

## Cross-Site Navigation

The navigation now properly connects all rbee properties:

| Link | URL | Purpose |
|------|-----|---------|
| **Logo** | https://rbee.dev | Homepage |
| **Home** | https://rbee.dev | Homepage |
| **Marketplace** | https://marketplace.rbee.dev | Browse models/workers |
| **Docs** | /docs | Documentation home |
| **Quick Start** | /docs/getting-started/installation | Installation guide |
| **API** | /docs/reference/api-openai-compatible | API reference |
| **Architecture** | /docs/architecture/overview | Architecture docs |
| **GitHub** | https://github.com/veighnsche/llama-orch | Source code |
| **Download** | /docs/getting-started/installation | Installation guide |

## Visual Verification

### Before (Issue)
- ❌ Page title "GPU Providers & Platforms" hidden behind nav
- ❌ Content starts at top of viewport
- ❌ No way to navigate to homepage or marketplace

### After (Fixed)
- ✅ Page title fully visible below nav
- ✅ Content starts below navigation bar
- ✅ Clear navigation to all rbee properties
- ✅ Logo links to homepage
- ✅ "Download" CTA in prominent position

## Build & Deploy

```bash
cd frontend/apps/user-docs
pnpm run build
# ✅ 35 static pages generated

npx wrangler pages deploy out/ --project-name=rbee-user-docs --branch=main
# ✅ 289 files uploaded (65 already uploaded)
# ✅ Deployed to https://main.rbee-user-docs.pages.dev
```

## Testing Checklist

- [x] Page content no longer hidden behind nav
- [x] Logo links to rbee.dev
- [x] "Home" link works
- [x] "Marketplace" link works
- [x] "Download" CTA links to installation
- [x] All internal doc links work
- [x] Mobile menu shows all links
- [x] Responsive padding (mobile vs desktop)

## Technical Details

### Navigation Height
The Navigation component uses:
```tsx
<nav className="fixed top-0 inset-x-0 z-50 ...">
  <div className="h-16 md:h-14"> {/* 4rem mobile, 3.5rem desktop */}
```

### Main Content Padding
Must match navigation height:
```css
#main {
  padding-top: 4rem;  /* matches h-16 */
}

@media (min-width: 768px) {
  #main {
    padding-top: 3.5rem;  /* matches h-14 */
  }
}
```

### Why Not Use Margin?
- Padding is better for fixed positioning
- Prevents margin collapse issues
- More predictable with Nextra's layout

## Related Issues

This fix follows the same pattern as:
- Marketplace navigation (already working)
- Commercial site navigation (if it exists)

All rbee apps should use:
1. Fixed navigation with `fixed top-0`
2. Main content with padding-top matching nav height
3. Cross-site navigation links

---

**TEAM-427 SIGNATURE:** User-docs navigation fixed - no more content overlap, proper cross-site links.
