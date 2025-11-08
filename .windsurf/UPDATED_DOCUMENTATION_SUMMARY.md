# Updated Documentation Summary

**TEAM-XXX: All documents updated with file renaming guidance**

## 🎉 What Changed

Added comprehensive guidance about renaming Props files from `.tsx` to `.ts` after removing JSX.

## 📚 Updated Documents

| Document | What Was Added |
|----------|----------------|
| **[HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md)** | Section on file extensions with explanation |
| **[EXAMPLE_TERMS_PAGE_MIGRATION.md](./EXAMPLE_TERMS_PAGE_MIGRATION.md)** | File rename step in changes summary + explanation |
| **[COMMERCIAL_APP_SSG_FIX_PLAN.md](./COMMERCIAL_APP_SSG_FIX_PLAN.md)** | Step 2: Rename Props File with rationale |
| **[ASIDES_SOLUTION_SUMMARY.md](./ASIDES_SOLUTION_SUMMARY.md)** | File rename in checklist + .ts in examples |
| **[ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md)** | Link to new FILE_RENAMING_GUIDE.md |

## 🆕 New Document

**[FILE_RENAMING_GUIDE.md](./FILE_RENAMING_GUIDE.md)** - Complete guide covering:
- Why rename .tsx → .ts
- When to rename
- How to rename (git mv, mv, IDE)
- Bulk rename script
- Verification steps
- Common issues and solutions
- Best practices

## 🎯 Key Points

### Why Rename?

**`.tsx` = TypeScript + JSX** (React components)  
**`.ts` = TypeScript only** (data/config/types)

After migration:
- ✅ Props files have NO JSX
- ✅ Props files only have config objects
- ✅ Props files only import types
- ✅ Should use `.ts` extension

### Benefits

| Benefit | Description |
|---------|-------------|
| **Clearer Intent** | `.ts` signals "data file" not "component" |
| **Faster Compilation** | No JSX parsing overhead |
| **Better IDE Performance** | Less syntax highlighting work |
| **Follows Conventions** | TypeScript best practices |

### When to Rename

Rename **AFTER** you've:
1. ✅ Removed all JSX from Props file
2. ✅ Removed all Lucide icon imports
3. ✅ Added `asideConfig` object
4. ✅ Verified build works
5. ✅ Tested page renders correctly

### How to Rename

```bash
# Recommended: Preserves git history
git mv PageProps.tsx PageProps.ts

# Or use IDE rename feature
# Most IDEs update imports automatically
```

## 📋 Migration Checklist (Updated)

Per page:

- [ ] Identify current aside content
- [ ] Choose appropriate variant
- [ ] Update Props.tsx with `asideConfig`
- [ ] Remove all JSX and Lucide imports
- [ ] **Rename Props.tsx → Props.ts**
- [ ] Update Page.tsx import (if needed)
- [ ] Update Page.tsx with `renderAside()`
- [ ] Test build
- [ ] Verify rendering

## 🔧 Bulk Rename Script

For renaming all Props files at once:

```bash
#!/bin/bash
cd frontend/apps/commercial/components/pages

find . -name "*Props.tsx" -not -name "*.bak" | while read file; do
  newfile="${file%.tsx}.ts"
  echo "Renaming: $file → $newfile"
  git mv "$file" "$newfile"
done
```

## 📊 Files to Rename

After migration, these should be `.ts`:

```
frontend/apps/commercial/components/pages/
├── TermsPage/TermsPageProps.tsx → .ts
├── PrivacyPage/PrivacyPageProps.tsx → .ts
├── RhaiScriptingPage/RhaiScriptingPageProps.tsx → .ts
├── DevelopersPage/DevelopersPageProps.tsx → .ts
├── ResearchPage/ResearchPageProps.tsx → .ts
├── HomelabPage/HomelabPageProps.tsx → .ts
├── CommunityPage/CommunityPageProps.tsx → .ts
├── EducationPage/EducationPageProps.tsx → .ts
├── ProvidersPage/ProvidersPageProps.tsx → .ts
└── StartupsPage/StartupsPageProps.tsx → .ts
```

## ✅ Verification

After renaming all files:

```bash
# 1. Check no .tsx Props files remain
find frontend/apps/commercial/components/pages -name "*Props.tsx" -not -name "*.bak"
# Should return nothing

# 2. Check all .ts Props files exist
find frontend/apps/commercial/components/pages -name "*Props.ts"
# Should list 10 files

# 3. Build succeeds
cd frontend/apps/commercial && pnpm build

# 4. Type check passes
pnpm run type-check
```

## 🎯 Example: Before & After

### Before Migration
```
TermsPageProps.tsx
├── Contains: <FileText className="..." />
├── Imports: import { FileText } from 'lucide-react'
└── Extension: .tsx ✅ (correct, has JSX)
```

### After Migration
```
TermsPageProps.ts
├── Contains: asideConfig: { variant: 'icon', icon: 'FileText', ... }
├── Imports: import type { AsideConfig } from '...'
└── Extension: .ts ✅ (correct, no JSX)
```

## 📞 Common Questions

**Q: Will imports break after renaming?**  
A: Usually no. TypeScript resolves both `.ts` and `.tsx` automatically. Only explicit imports with extensions need updating.

**Q: Should I rename before or after removing JSX?**  
A: **AFTER!** Remove JSX first, verify it works, then rename.

**Q: Can I use regular `mv` instead of `git mv`?**  
A: Yes, but `git mv` preserves file history which is better for git blame/log.

**Q: Do I need to update import statements?**  
A: Usually no, unless you have explicit `.tsx` extensions in imports (rare).

## 🎉 Summary

All documentation has been updated to include:
- ✅ File renaming guidance
- ✅ Explanation of .tsx vs .ts
- ✅ Step-by-step instructions
- ✅ Bulk rename script
- ✅ Verification steps
- ✅ Common issues and solutions

**Next Steps:**
1. Read [FILE_RENAMING_GUIDE.md](./FILE_RENAMING_GUIDE.md)
2. Migrate pages following updated guides
3. Rename Props files after removing JSX
4. Verify all builds succeed

---

**Updated:** 2025-11-08  
**Status:** All documentation updated  
**New Guide:** FILE_RENAMING_GUIDE.md created  
**Ready:** For execution with file renaming
