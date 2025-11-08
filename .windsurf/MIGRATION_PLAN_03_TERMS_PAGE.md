# Step 3: Migrate TermsPage

**Phase:** 1 - High Priority  
**Time:** 15 minutes  
**Priority:** HIGH  
**Variant:** IconAside

## 🎯 Goal

Migrate TermsPage to use IconAside component configured from Props file.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/TermsPage/
├── TermsPageProps.tsx  → Will rename to .ts
└── TermsPage.tsx       → Update imports and aside
```

## 🔍 Current State Analysis

```bash
cd frontend/apps/commercial/components/pages/TermsPage

# Check current Props file
cat TermsPageProps.tsx | grep -A 10 "aside"

# Check current Page file
cat TermsPage.tsx | grep -A 5 "aside"
```

## ✏️ Changes Needed

### 1. Update TermsPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Remove:**
```typescript
aside: null as any,
// or
aside: (<div>...</div>)
```

**Add:**
```typescript
asideConfig: {
  variant: 'icon',
  icon: 'FileText',
  title: 'Legal Document',
  subtitle: 'Please read carefully'
} as AsideConfig
```

**Remove Lucide imports if present:**
```typescript
// Remove this line if it exists:
import { FileText, Scale } from 'lucide-react'
```

### 2. Update TermsPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Remove Lucide imports if present:**
```typescript
// Remove if exists:
import { FileText } from 'lucide-react'
```

**Update HeroTemplate usage:**
```typescript
// BEFORE:
<HeroTemplate
  {...termsHeroProps}
  aside={<div>...JSX...</div>}
/>

// AFTER:
<HeroTemplate
  {...termsHeroProps}
  aside={renderAside(termsHeroProps.asideConfig)}
/>
```

### 3. Rename Props File

```bash
# After removing all JSX
git mv TermsPageProps.tsx TermsPageProps.ts
```

## 📝 Implementation Steps

```bash
cd frontend/apps/commercial/components/pages/TermsPage

# 1. Backup current files
cp TermsPageProps.tsx TermsPageProps.tsx.bak
cp TermsPage.tsx TermsPage.tsx.bak

# 2. Edit TermsPageProps.tsx
# - Add AsideConfig import
# - Remove aside prop
# - Add asideConfig object
# - Remove Lucide imports

# 3. Edit TermsPage.tsx
# - Add renderAside import
# - Remove Lucide imports
# - Update aside prop to use renderAside()

# 4. Test compilation
cd ../../..
pnpm run type-check

# 5. If successful, rename Props file
cd components/pages/TermsPage
git mv TermsPageProps.tsx TermsPageProps.ts

# 6. Update import in TermsPage.tsx if needed
# Usually not needed - TypeScript resolves both

# 7. Test again
cd ../../..
pnpm run type-check

# 8. Test build
pnpm build

# 9. If successful, remove backups
cd components/pages/TermsPage
rm *.bak
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# 1. Check Props file has no JSX
! grep -q "<" components/pages/TermsPage/TermsPageProps.ts && echo "✅ No JSX" || echo "❌ Still has JSX"

# 2. Check Props file has asideConfig
grep -q "asideConfig" components/pages/TermsPage/TermsPageProps.ts && echo "✅ Has asideConfig" || echo "❌ Missing asideConfig"

# 3. Check Props file has no Lucide imports
! grep -q "from 'lucide-react'" components/pages/TermsPage/TermsPageProps.ts && echo "✅ No Lucide" || echo "❌ Still has Lucide"

# 4. Check Page file uses renderAside
grep -q "renderAside" components/pages/TermsPage/TermsPage.tsx && echo "✅ Uses renderAside" || echo "❌ Missing renderAside"

# 5. Check file extension
test -f components/pages/TermsPage/TermsPageProps.ts && echo "✅ Renamed to .ts" || echo "⚠️  Still .tsx"

# 6. Type check
pnpm run type-check 2>&1 | grep -i "termspage" || echo "✅ No TypeScript errors"

# 7. Build test
pnpm build 2>&1 | grep -i "termspage" || echo "✅ Build succeeds"
```

## 📋 Checklist

- [ ] TermsPageProps.tsx backed up
- [ ] TermsPage.tsx backed up
- [ ] Added AsideConfig import to Props
- [ ] Removed aside prop from Props
- [ ] Added asideConfig object to Props
- [ ] Removed Lucide imports from Props
- [ ] Added renderAside import to Page
- [ ] Updated aside prop in Page to use renderAside()
- [ ] Removed Lucide imports from Page
- [ ] Type check passes
- [ ] Renamed TermsPageProps.tsx → TermsPageProps.ts
- [ ] Type check still passes
- [ ] Build succeeds
- [ ] Removed backup files

## 🎨 Expected Result

### TermsPageProps.ts (Final)
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
import type { HeroTemplateProps } from '@rbee/ui/templates'

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
  asideConfig: {
    variant: 'icon',
    icon: 'FileText',
    title: 'Legal Document',
    subtitle: 'Please read carefully'
  } as AsideConfig
}
```

### TermsPage.tsx (Final)
```typescript
import { renderAside } from '../../organisms/HeroAsides'
import { termsHeroProps } from './TermsPageProps'
// ... other imports

export default function TermsPage() {
  return (
    <HeroTemplate
      {...termsHeroProps}
      aside={renderAside(termsHeroProps.asideConfig)}
    />
    {/* ... rest of page */}
  )
}
```

## ✅ Success Criteria

- ✅ Props file is .ts (not .tsx)
- ✅ Props file has no JSX
- ✅ Props file has asideConfig
- ✅ Props file has no Lucide imports
- ✅ Page uses renderAside()
- ✅ Type check passes
- ✅ Build succeeds
- ✅ Page renders correctly

## 🚀 Next Step

Once complete, proceed to:
**[STEP_04_PRIVACY_PAGE.md](./MIGRATION_PLAN_04_PRIVACY_PAGE.md)** - Migrate PrivacyPage

---

**Status:** Implementation step  
**Blocking:** No - can skip if needed  
**Time:** 15 minutes  
**Difficulty:** Easy (first example)
