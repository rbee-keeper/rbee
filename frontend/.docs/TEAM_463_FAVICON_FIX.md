# TEAM-463: Favicon Fix Across All Apps

**Date:** 2025-11-10  
**Author:** TEAM-463  
**Status:** ✅ COMPLETE

## Problem

Favicons were not being picked up by Next.js 13+ App Router across all frontend apps because the `icons` metadata was missing from the root layout files.

## Root Cause

Next.js 13+ App Router requires **explicit icon metadata** in `layout.tsx`. Simply having `favicon.ico` in the `public/` folder is not enough - you must declare it in the metadata.

## Solution

### 1. Created Red Favicon for Admin App

**File:** `/frontend/apps/admin/public/favicon.svg`

Created a red-themed bee icon to distinguish the admin dashboard:
- **Body color**: `#ef4444` (red-500)
- **Stripes/stinger**: `#7f1d1d` (red-900)
- **Wings**: `#ef4444` with 0.7 opacity

**Conversion to ICO:**
```bash
cd frontend/apps/admin/public
convert -background none -define icon:auto-resize=256,128,96,64,48,32,16 favicon.svg favicon.ico
```

### 2. Added Icons Metadata to All Layouts

Updated all 4 app layouts to include explicit favicon declarations:

#### Commercial (`apps/commercial/app/layout.tsx`)
```typescript
icons: {
  icon: [
    { url: '/favicon.ico', sizes: 'any' },
    { url: '/favicon.svg', type: 'image/svg+xml' },
  ],
  apple: '/favicon-192x192.png',
},
```
**Color:** 🟠 Orange (`#f59e0b`)

#### Admin (`apps/admin/app/layout.tsx`)
```typescript
icons: {
  icon: [
    { url: '/favicon.ico', sizes: 'any' },
    { url: '/favicon.svg', type: 'image/svg+xml' },
  ],
},
```
**Color:** 🔴 Red (`#ef4444`)

#### Marketplace (`apps/marketplace/app/layout.tsx`)
```typescript
icons: {
  icon: [
    { url: '/favicon.ico', sizes: 'any' },
    { url: '/favicon.svg', type: 'image/svg+xml' },
  ],
},
```
**Color:** 🟣 Purple (`#a855f7`)

#### User Docs (`apps/user-docs/app/layout.tsx`)
```typescript
icons: {
  icon: [
    { url: '/favicon.ico', sizes: 'any' },
    { url: '/favicon.svg', type: 'image/svg+xml' },
  ],
},
```
**Color:** ⚫ Slate (`#64748b`)

## Favicon Color Scheme

Each app now has a unique color to help users distinguish between tabs:

| App | Color | Hex | Purpose |
|-----|-------|-----|---------|
| **Commercial** | 🟠 Orange | `#f59e0b` | Main marketing site |
| **Admin** | 🔴 Red | `#ef4444` | Admin dashboard (alerts/warnings) |
| **Marketplace** | 🟣 Purple | `#a855f7` | Model marketplace |
| **User Docs** | ⚫ Slate | `#64748b` | Documentation |

## Files Modified

1. ✅ **Created:** `apps/admin/public/favicon.svg` (red theme)
2. ✅ **Created:** `apps/admin/public/favicon.ico` (converted from SVG)
3. ✅ **Updated:** `apps/commercial/app/layout.tsx` (added icons metadata)
4. ✅ **Updated:** `apps/admin/app/layout.tsx` (added icons metadata)
5. ✅ **Updated:** `apps/marketplace/app/layout.tsx` (added icons metadata)
6. ✅ **Updated:** `apps/user-docs/app/layout.tsx` (added icons metadata)

## How Next.js Icons Work

### Icon Priority (Next.js 13+)

1. **SVG** - Modern browsers prefer SVG (scalable, crisp)
2. **ICO** - Fallback for older browsers
3. **PNG** - Apple devices (apple-touch-icon)

### Metadata Structure

```typescript
icons: {
  icon: [
    { url: '/favicon.ico', sizes: 'any' },      // Fallback for all sizes
    { url: '/favicon.svg', type: 'image/svg+xml' }, // Modern browsers
  ],
  apple: '/favicon-192x192.png',  // iOS/macOS (optional)
}
```

## Verification

After deployment, each app will show its unique colored bee icon in:
- Browser tabs
- Bookmarks
- History
- Mobile home screen (if added)

## Why This Matters

**User Experience:**
- Users can quickly identify which rbee app they're on
- Multiple tabs are easier to distinguish
- Professional branding consistency

**Technical:**
- Proper SEO metadata
- PWA-ready (if needed later)
- Cross-browser compatibility

---

**Result:** All 4 frontend apps now have unique, properly configured favicons that will display correctly in all browsers! 🎉
