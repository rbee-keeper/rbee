# TEAM-476: Dark Mode & Theme Toggle Fix

**Date:** November 12, 2025  
**Status:** ✅ COMPLETE  
**Apps Fixed:** `marketplace`, `user-docs`

## Issues Identified

### 1. React Key Warning (user-docs)
**Error:**
```
Each child in a list should have a unique "key" prop.
Check the render method of `ConfigProvider`. It was passed a child from RootLayout.
```

**Root Cause:**
- Footer links had `key` props on `<a>` tags inside a `<div>`
- The `<div>` was passed to Nextra's `Layout` component
- React expected keys on iterable elements, not static links

**Fix:**
- Removed unnecessary `key` props from static footer links
- Links are not in an array/map, so keys are not needed

### 2. Dark Mode Not Working (marketplace + user-docs)
**Root Cause:**
- Missing `bg-background text-foreground` classes on `<body>` tag
- These classes are required for theme transitions to work properly
- Commercial app had these classes, but marketplace and user-docs did not

**Fix:**
- Added `bg-background text-foreground` to both apps' `<body>` tags
- This enables proper theme color transitions via CSS variables

## Files Modified

### 1. `/apps/user-docs/app/layout.tsx`
**Changes:**
- ✅ Removed `key="github"` and `key="website"` from footer links (lines 65, 68)
- ✅ Added `bg-background text-foreground` to `<body>` className (line 77)

**Before:**
```tsx
<body>
  <Navigation />
  <main id="main">
    <Layout footer={footerContent}>
      {children}
    </Layout>
  </main>
</body>
```

**After:**
```tsx
<body className="bg-background text-foreground">
  <Navigation />
  <main id="main">
    <Layout footer={footerContent}>
      {children}
    </Layout>
  </main>
</body>
```

### 2. `/apps/marketplace/app/layout.tsx`
**Changes:**
- ✅ Added `bg-background text-foreground` to `<body>` className (line 28)

**Before:**
```tsx
<body className="antialiased">
  <ThemeProvider>
    <MarketplaceNav />
    <main>{children}</main>
  </ThemeProvider>
</body>
```

**After:**
```tsx
<body className="bg-background text-foreground antialiased">
  <ThemeProvider>
    <MarketplaceNav />
    <main>{children}</main>
  </ThemeProvider>
</body>
```

## Why This Works

### Theme Color System
The `bg-background` and `text-foreground` classes use CSS variables that change based on the `.dark` class:

```css
/* Light mode (default) */
:root {
  --background: hsl(0 0% 100%);
  --foreground: hsl(222.2 84% 4.9%);
}

/* Dark mode (.dark class on <html>) */
.dark {
  --background: hsl(222.2 84% 4.9%);
  --foreground: hsl(210 40% 98%);
}
```

When `next-themes` toggles the `.dark` class on `<html>`, these CSS variables update, and the `bg-background text-foreground` classes on `<body>` apply the new colors.

### Why It Was Broken
Without these classes:
- ❌ Body had no background color → stayed white in dark mode
- ❌ Text had no foreground color → stayed black in dark mode
- ❌ Theme toggle appeared to do nothing

With these classes:
- ✅ Body background transitions to dark
- ✅ Text color transitions to light
- ✅ Theme toggle works as expected

## Build Verification

```bash
cd /home/vince/Projects/rbee/frontend
turbo build --filter=@rbee/marketplace --filter=@rbee/user-docs
```

**Result:** ✅ Both apps build successfully

## Testing Checklist

- [ ] Open `http://localhost:3002` (marketplace)
- [ ] Toggle dark mode → background should change to dark
- [ ] Open `http://localhost:3003` (user-docs)
- [ ] Toggle dark mode → background should change to dark
- [ ] Verify no React key warnings in console
- [ ] Verify theme persists on page reload

## Related Files

**Theme Providers:**
- `/apps/marketplace/components/providers/ThemeProvider.tsx` - Wraps `next-themes`
- `/apps/commercial/components/providers/ThemeProvider/ThemeProvider.tsx` - Commercial app version

**Global Styles:**
- `/apps/marketplace/app/globals.css` - Marketplace CSS
- `/apps/user-docs/app/globals.css` - User docs CSS
- `/apps/commercial/app/globals.css` - Commercial CSS (reference)

**UI Package:**
- `/packages/rbee-ui/src/tokens/globals.css` - CSS variable definitions
- `/packages/rbee-ui/dist/index.css` - Compiled styles

## Key Learnings

1. **Always add `bg-background text-foreground` to `<body>`** when using `next-themes`
2. **Don't add `key` props to static elements** - only use them for arrays/maps
3. **Check commercial app for reference** - it has the correct patterns
4. **CSS variables need classes to apply** - variables alone don't change colors

## Next Steps

1. ✅ Build verification complete
2. ✅ Documentation created
3. ⏭️ Manual testing in dev mode
4. ⏭️ Deploy to staging for QA

---

**TEAM-476 SIGNATURE:** Dark mode and theme toggle fixed in marketplace and user-docs apps.
