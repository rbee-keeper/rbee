# Step 4: Migrate PrivacyPage

**Phase:** 1 - High Priority  
**Time:** 15 minutes  
**Priority:** HIGH  
**Variant:** IconAside

## 🎯 Goal

Migrate PrivacyPage to use IconAside component (similar to TermsPage).

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/PrivacyPage/
├── PrivacyPageProps.tsx  → Will rename to .ts
└── PrivacyPage.tsx       → Update imports and aside
```

## ✏️ Changes Needed

### 1. Update PrivacyPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'icon',
  icon: 'Shield',  // Privacy/security icon
  title: 'Privacy Policy',
  subtitle: 'Your data is protected'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update PrivacyPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...privacyHeroProps}
  aside={renderAside(privacyHeroProps.asideConfig)}
/>
```

**Remove Lucide imports if present.**

### 3. Rename Props File

```bash
git mv PrivacyPageProps.tsx PrivacyPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial/components/pages/PrivacyPage

# 1. Backup
cp PrivacyPageProps.tsx PrivacyPageProps.tsx.bak
cp PrivacyPage.tsx PrivacyPage.tsx.bak

# 2. Edit files (follow TermsPage pattern)

# 3. Test
cd ../../..
pnpm run type-check

# 4. Rename
cd components/pages/PrivacyPage
git mv PrivacyPageProps.tsx PrivacyPageProps.ts

# 5. Verify
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Quick checks
! grep -q "<" components/pages/PrivacyPage/PrivacyPageProps.ts && echo "✅ No JSX"
grep -q "asideConfig" components/pages/PrivacyPage/PrivacyPageProps.ts && echo "✅ Has asideConfig"
grep -q "renderAside" components/pages/PrivacyPage/PrivacyPage.tsx && echo "✅ Uses renderAside"
test -f components/pages/PrivacyPage/PrivacyPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with Shield icon added
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 Icon Options

Choose the best icon for privacy:
- `'Shield'` - Security/protection (recommended)
- `'Lock'` - Privacy/locked
- `'Eye'` - Visibility/privacy
- `'FileText'` - Document

## 🚀 Next Step

**[STEP_05_RHAI_SCRIPTING_PAGE.md](./MIGRATION_PLAN_05_RHAI_SCRIPTING_PAGE.md)** - Migrate RhaiScriptingPage

---

**Time:** 15 minutes  
**Difficulty:** Easy (same as TermsPage)
