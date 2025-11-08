# Example: TermsPage Migration

**TEAM-XXX: Step-by-step example of migrating TermsPage to use reusable asides**

## Current State (Broken)

### TermsPageProps.tsx
```tsx
export const termsHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    text: 'Legal • Terms of Service',
    icon: 'Scale',
  },
  headline: {
    variant: 'simple',
    content: 'Terms of Service',
  },
  subcopy: 'Transparent terms. No hidden clauses. Last updated: October 17, 2025.',
  // ... other props
  aside: null as any, // BROKEN - was JSX, now null
}
```

### TermsPage.tsx
```tsx
import { FileText, Scale } from 'lucide-react'

<HeroTemplate
  {...termsHeroProps}
  aside={
    <div className="flex items-center justify-center rounded border border-border bg-card/60 backdrop-blur-sm p-8 shadow-sm">
      <div className="text-center space-y-4">
        <FileText className="h-16 w-16 mx-auto text-muted-foreground" />
        <div className="space-y-2">
          <p className="text-sm font-medium text-card-foreground">Legal Document</p>
          <p className="text-xs text-muted-foreground">Please read carefully</p>
        </div>
      </div>
    </div>
  }
/>
```

## New State (Fixed + CMS-Friendly)

### TermsPageProps.ts ← Note: Renamed from `.tsx` to `.ts`
```tsx
import type { AsideConfig } from '../../organisms/HeroAsides'

export const termsHeroProps: HeroTemplateProps = {
  badge: {
    variant: 'icon',
    text: 'Legal • Terms of Service',
    icon: 'Scale',
  },
  headline: {
    variant: 'simple',
    content: 'Terms of Service',
  },
  subcopy: 'Transparent terms. No hidden clauses. Last updated: October 17, 2025.',
  // ... other props
  asideConfig: {
    variant: 'icon',
    icon: 'FileText',
    title: 'Legal Document',
    subtitle: 'Please read carefully'
  } as AsideConfig
}
```

### TermsPage.tsx
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate
  {...termsHeroProps}
  aside={renderAside(termsHeroProps.asideConfig)}
/>
```

## Changes Summary

### TermsPageProps.ts Changes (renamed from .tsx)

1. **Rename file:**
   ```bash
   mv TermsPageProps.tsx TermsPageProps.ts
   ```

2. **Add import:**
   ```tsx
   import type { AsideConfig } from '../../organisms/HeroAsides'
   ```

2. **Remove:**
   ```tsx
   aside: null as any,
   ```

3. **Add:**
   ```tsx
   asideConfig: {
     variant: 'icon',
     icon: 'FileText',
     title: 'Legal Document',
     subtitle: 'Please read carefully'
   } as AsideConfig
   ```

### TermsPage.tsx Changes

1. **Remove imports:**
   ```tsx
   import { FileText, Scale } from 'lucide-react'
   ```

2. **Add import:**
   ```tsx
   import { renderAside } from '../../organisms/HeroAsides'
   ```

3. **Replace aside prop:**
   ```tsx
   // BEFORE:
   aside={<div>...long JSX...</div>}
   
   // AFTER:
   aside={renderAside(termsHeroProps.asideConfig)}
   ```

## Why Rename to .ts?

**`.tsx` vs `.ts`:**
- `.tsx` = TypeScript + JSX (React components)
- `.ts` = TypeScript only (no JSX)

**Props files after migration:**
- ✅ Only contain config objects (no JSX)
- ✅ Only import types (no components)
- ✅ Are pure data/configuration
- ✅ Should use `.ts` extension

**Benefits of correct extension:**
- Clearer intent (data file, not component)
- Faster TypeScript compilation (no JSX parsing)
- Better IDE performance
- Follows TypeScript conventions

## Benefits

✅ **Props.ts is now CMS:**
- All content in one place
- Easy to edit: change icon, title, subtitle
- No code knowledge needed

✅ **No JSX serialization:**
- `asideConfig` is a plain object
- Serializable for SSG
- No build errors

✅ **Reusable:**
- IconAside component handles all icon asides
- Consistent styling across pages
- Easy to maintain

✅ **Type-safe:**
- `AsideConfig` union type
- TypeScript catches errors
- IntelliSense support

## Testing

```bash
# Build commercial app
cd frontend/apps/commercial && pnpm build

# Should succeed with no errors
# Page should render with aside visible
```

## Next Pages

Use the same pattern for:
- PrivacyPage (icon aside with Shield)
- Other pages with icon asides

For pages with images or stats, use:
- `variant: 'image'` with image config
- `variant: 'stats'` with stats array
- `variant: 'card'` with custom content
