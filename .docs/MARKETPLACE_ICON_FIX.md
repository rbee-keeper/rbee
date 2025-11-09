# Marketplace Icon Fix - Complete Guide

## Problem
Icons (Zap, Search, Database) are not rendering in the FeatureInfoCard components on the marketplace homepage.

## Root Cause
The marketplace is passing icon components as **component references** instead of **JSX elements**:

```tsx
// ❌ WRONG - Component reference
<FeatureInfoCard icon={Zap} title="..." />

// ✅ CORRECT - JSX element
<FeatureInfoCard icon={<Zap />} title="..." />
```

## Why This Matters
The `FeatureInfoCard` component (lines 196-215) tries to handle both:
1. **Component references** (`typeof icon === 'function'`)
2. **JSX elements** (`React.isValidElement(icon)`)

However, in SSR/SSG builds with OpenNext on Cloudflare Workers, the function type check may fail due to:
- Webpack bundling transformations
- React Server Components serialization
- Cloudflare Workers runtime differences

## The Fix

### Option 1: Change Marketplace to Use JSX Elements (RECOMMENDED)
**File:** `/home/vince/Projects/rbee/frontend/apps/marketplace/app/page.tsx`

```tsx
// Change lines 44, 52, 60 from:
icon={Zap}
icon={Search}
icon={Database}

// To:
icon={<Zap />}
icon={<Search />}
icon={<Database />}
```

**Why this works:**
- Storybook examples all use JSX elements
- `React.isValidElement()` is more reliable in SSR/SSG
- Matches the documented API in FeatureInfoCard.stories.tsx

### Option 2: Fix FeatureInfoCard to Handle Component References Better
**File:** `/home/vince/Projects/rbee/frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/FeatureInfoCard.tsx`

```tsx
// Replace lines 196-215 with:
export function FeatureInfoCard({
  icon,
  title,
  body,
  tag,
  tone,
  size,
  className,
  delay,
  showBorder,
  variant = 'default',
}: FeatureInfoCardProps) {
  // Render icon - handle both Component and JSX
  const renderIcon = () => {
    // If it's already a JSX element, clone it with className
    if (React.isValidElement(icon)) {
      return React.cloneElement(icon, {
        // @ts-expect-error - icon className merging
        className: cn(icon.props.className, iconVariants({ tone, variant })),
      })
    }
    
    // If it's a component reference, render it
    if (typeof icon === 'function') {
      const IconComponent = icon as React.ComponentType<{ className?: string }>
      return <IconComponent className={iconVariants({ tone, variant })} />
    }
    
    // Fallback: render as-is
    return icon
  }

  return (
    <Card className={cn(featureInfoCardVariants({ tone, showBorder }), delay, className)}>
      <CardContent className={contentPaddingVariants({ variant })}>
        {/* Icon */}
        <div className={iconContainerVariants({ tone, variant })} aria-hidden="true">
          {renderIcon()}
        </div>

        {/* Title */}
        <h3 className={titleVariants({ variant })}>{title}</h3>

        {/* Body */}
        <p className={bodyVariants({ size, variant })}>{body}</p>

        {/* Optional Tag */}
        {tag && <span className={tagVariants({ tone })}>{tag}</span>}
      </CardContent>
    </Card>
  )
}
```

## Recommended Approach

**Use Option 1** - it's simpler and matches the documented API.

1. Edit `/home/vince/Projects/rbee/frontend/apps/marketplace/app/page.tsx`
2. Change lines 44, 52, 60 to use JSX elements
3. Rebuild and redeploy: `pnpm run deploy`
4. Verify icons appear

## Verification Steps

1. **Local dev server:**
   ```bash
   cd /home/vince/Projects/rbee/frontend/apps/marketplace
   pnpm dev
   ```
   Open http://localhost:7823 and check homepage

2. **Production build:**
   ```bash
   pnpm run deploy
   ```
   Check https://rbee-marketplace.vpdl.workers.dev/

3. **Visual check:**
   - Homepage should show 3 feature cards
   - Each card should have a colored icon container
   - Icons should be visible: ⚡ (Zap), 🔍 (Search), 💾 (Database)

## Why Icons Might Still Not Show

If icons still don't appear after Option 1:

1. **Lucide React not bundled properly**
   - Check if `lucide-react` is in `package.json` dependencies
   - Verify it's not in `devDependencies`

2. **CSS not loaded**
   - Check browser console for CSS 404 errors
   - Verify `.open-next/assets` contains CSS files
   - Check `wrangler.jsonc` has `assets` binding

3. **Icon size is 0**
   - Check if Tailwind classes are being purged
   - Verify `h-6 w-6` classes exist in final CSS

4. **React hydration mismatch**
   - Check browser console for hydration errors
   - Icons might be rendering but getting removed by React

## Debug Commands

```bash
# Check if lucide-react is installed
cd /home/vince/Projects/rbee/frontend/apps/marketplace
pnpm list lucide-react

# Check built assets
ls -lh .open-next/assets/_next/static/css/

# Check if icons are in the bundle
grep -r "lucide-react" .next/

# Deploy and check
pnpm run deploy
```

## Related Files

- `/home/vince/Projects/rbee/frontend/apps/marketplace/app/page.tsx` - Homepage using icons
- `/home/vince/Projects/rbee/frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/FeatureInfoCard.tsx` - Component
- `/home/vince/Projects/rbee/frontend/packages/rbee-ui/src/molecules/FeatureInfoCard/FeatureInfoCard.stories.tsx` - Examples

## Success Criteria

✅ Icons visible on homepage  
✅ Icons have correct colors (primary theme)  
✅ Icons are properly sized (h-6 w-6)  
✅ No console errors  
✅ Works in both dev and production  
