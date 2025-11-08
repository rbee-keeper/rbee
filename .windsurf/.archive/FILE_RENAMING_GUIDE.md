# File Renaming Guide: .tsx → .ts

**TEAM-XXX: Rename Props files after removing JSX**

## Why Rename?

**`.tsx` vs `.ts`:**
- `.tsx` = TypeScript + JSX (React components, can use `<Component />`)
- `.ts` = TypeScript only (no JSX, pure data/config/types)

**After migration:**
- Props files contain NO JSX
- Props files only have config objects
- Props files only import types
- Props files should use `.ts` extension

## Benefits of Correct Extension

| Benefit | Description |
|---------|-------------|
| **Clearer Intent** | `.ts` signals "data file" not "component file" |
| **Faster Compilation** | TypeScript doesn't parse for JSX |
| **Better IDE Performance** | Less syntax highlighting overhead |
| **Follows Conventions** | TypeScript best practices |
| **Prevents Confusion** | Clear separation: data vs components |

## When to Rename

Rename Props.tsx → Props.ts **AFTER** you've:

1. ✅ Removed all JSX from the file
2. ✅ Removed all Lucide icon imports
3. ✅ Added `asideConfig` object
4. ✅ Verified build works with new config
5. ✅ Tested the page renders correctly

## How to Rename

### Option 1: Git mv (Preserves History)

```bash
# Recommended: Use git mv to preserve file history
git mv PageProps.tsx PageProps.ts
```

### Option 2: Regular mv

```bash
# Alternative: Regular move
mv PageProps.tsx PageProps.ts
```

### Option 3: IDE Rename

Most IDEs (VS Code, WebStorm) have "Rename File" that updates imports automatically.

## Files to Rename

After migration, these files should be `.ts`:

```bash
# All Props files in commercial app
frontend/apps/commercial/components/pages/*/
  TermsPageProps.tsx → TermsPageProps.ts
  PrivacyPageProps.tsx → PrivacyPageProps.ts
  RhaiScriptingPageProps.tsx → RhaiScriptingPageProps.ts
  DevelopersPageProps.tsx → DevelopersPageProps.ts
  ResearchPageProps.tsx → ResearchPageProps.ts
  HomelabPageProps.tsx → HomelabPageProps.ts
  CommunityPageProps.tsx → CommunityPageProps.ts
  EducationPageProps.tsx → EducationPageProps.ts
  ProvidersPageProps.tsx → ProvidersPageProps.ts
  StartupsPageProps.tsx → StartupsPageProps.ts
```

## Bulk Rename Script

```bash
#!/bin/bash
# Rename all Props.tsx files to Props.ts in commercial app

cd frontend/apps/commercial/components/pages

# Find and rename all *Props.tsx files
find . -name "*Props.tsx" -not -name "*.bak" | while read file; do
  newfile="${file%.tsx}.ts"
  echo "Renaming: $file → $newfile"
  git mv "$file" "$newfile"
done

echo "Done! Don't forget to update imports if needed."
```

## Import Path Updates

**Usually NOT needed** - TypeScript resolves both `.ts` and `.tsx` automatically.

But if you have explicit imports with extensions:

```typescript
// BEFORE (if you had explicit extension):
import { heroProps } from './PageProps.tsx'

// AFTER:
import { heroProps } from './PageProps.ts'

// BEST (no extension - works with both):
import { heroProps } from './PageProps'
```

## Verification

After renaming:

```bash
# 1. Check TypeScript compilation
cd frontend/apps/commercial
pnpm run type-check

# 2. Check build
pnpm build

# 3. Check no .tsx Props files remain
find components/pages -name "*Props.tsx" -not -name "*.bak"
# Should return nothing

# 4. Check all .ts Props files exist
find components/pages -name "*Props.ts"
# Should list all renamed files
```

## Checklist

Per file:

- [ ] File contains NO JSX
- [ ] File contains NO Lucide imports
- [ ] File only has config objects and types
- [ ] Build succeeds with current name
- [ ] Rename .tsx → .ts
- [ ] Build still succeeds
- [ ] Page renders correctly
- [ ] Git commit the rename

## Common Issues

### Issue 1: Import Errors After Rename

**Symptom:** `Cannot find module './PageProps'`

**Solution:** 
- Check if import has explicit `.tsx` extension
- Remove extension or change to `.ts`
- Or let TypeScript resolve automatically (no extension)

### Issue 2: Build Fails After Rename

**Symptom:** Build errors after renaming

**Solution:**
- Verify file actually has no JSX
- Check for missed Lucide imports
- Clear build cache: `rm -rf .next && pnpm build`

### Issue 3: Git Shows as Delete + Add

**Symptom:** Git shows file as deleted and new file added

**Solution:**
- Use `git mv` instead of `mv`
- Or tell git it's a rename: `git add -A`
- Git should detect rename if >50% similarity

## Example: TermsPage

### Before
```
TermsPageProps.tsx (contains JSX)
├── import { FileText } from 'lucide-react'
├── aside: (<div><FileText />...</div>)
└── Extension: .tsx (correct, has JSX)
```

### After Migration
```
TermsPageProps.ts (no JSX)
├── import type { AsideConfig } from '...'
├── asideConfig: { variant: 'icon', icon: 'FileText', ... }
└── Extension: .ts (correct, no JSX)
```

### Rename Command
```bash
cd frontend/apps/commercial/components/pages/TermsPage
git mv TermsPageProps.tsx TermsPageProps.ts
```

## Best Practices

1. **Rename AFTER migration** - Don't rename before removing JSX
2. **Use git mv** - Preserves file history
3. **Test before commit** - Verify build and rendering
4. **Commit rename separately** - Easier to review
5. **Update docs** - Note the rename in commit message

## Commit Message Template

```
refactor(commercial): rename Props files .tsx → .ts

Props files no longer contain JSX after aside component migration.
Renamed to .ts extension following TypeScript conventions.

Files renamed:
- TermsPageProps.tsx → TermsPageProps.ts
- PrivacyPageProps.tsx → PrivacyPageProps.ts
- [... list all renamed files ...]

No functional changes, only file extension updates.
```

## Summary

| Before | After | Why |
|--------|-------|-----|
| `PageProps.tsx` | `PageProps.ts` | No JSX in file |
| Has `<Component />` | Has `{ config }` | Only data/config |
| Imports components | Imports types | Only type imports |
| `.tsx` extension | `.ts` extension | Correct convention |

---

**Remember:** Rename AFTER removing JSX, not before!
