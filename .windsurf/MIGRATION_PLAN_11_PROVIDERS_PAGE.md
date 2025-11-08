# Step 11: Migrate ProvidersPage

**Phase:** 3 - Low Priority  
**Time:** 10 minutes  
**Priority:** LOW  
**Variant:** ImageAside

## 🎯 Goal

Migrate ProvidersPage to use ImageAside component with existing GPU earnings image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/ProvidersPage/
├── ProvidersPageProps.tsx  → Will rename to .ts
└── ProvidersPage.tsx       → Update imports and aside
```

## 🖼️ Image Available

✅ **Existing image:** `gpu-earnings.png`

Location: `frontend/apps/commercial/public/images/gpu-earnings.png`

## ✏️ Changes Needed

### 1. Update ProvidersPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/gpu-earnings.png',
  alt: 'GPU earnings calculator showing revenue potential',
  width: 1024,   // Adjust based on actual image
  height: 1024,  // Adjust based on actual image
  title: 'Earn with Your GPUs',
  subtitle: 'Turn idle hardware into income'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update ProvidersPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...providersHeroProps}
  aside={renderAside(providersHeroProps.asideConfig)}
/>
```

### 3. Rename Props File

```bash
git mv ProvidersPageProps.tsx ProvidersPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Check image exists and dimensions
test -f public/images/gpu-earnings.png && echo "✅ Image exists"
file public/images/gpu-earnings.png  # Check dimensions

# 2. Backup files
cd components/pages/ProvidersPage
cp ProvidersPageProps.tsx ProvidersPageProps.tsx.bak
cp ProvidersPage.tsx ProvidersPage.tsx.bak

# 3. Edit files
# Adjust width/height based on actual dimensions

# 4. Test
cd ../../..
pnpm run type-check

# 5. Rename
cd components/pages/ProvidersPage
git mv ProvidersPageProps.tsx ProvidersPageProps.ts

# 6. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image
test -f public/images/gpu-earnings.png && echo "✅ Image exists"
file public/images/gpu-earnings.png

# Check Props file
grep -q "variant: 'image'" components/pages/ProvidersPage/ProvidersPageProps.ts && echo "✅ Uses ImageAside"
grep -q "gpu-earnings" components/pages/ProvidersPage/ProvidersPageProps.ts && echo "✅ Correct image"

# Check Page file
grep -q "renderAside" components/pages/ProvidersPage/ProvidersPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/ProvidersPage/ProvidersPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Image exists at `/images/gpu-earnings.png`
- [ ] Image dimensions checked
- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with image variant added
- [ ] Correct image dimensions specified
- [ ] Descriptive alt text added
- [ ] Title and subtitle added
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 Provider Theme

**Focus:** GPU earnings, revenue, monetization  
**Tone:** Business-focused, opportunity  
**Alt text:** Describe the calculator/earnings visualization

## 💡 Title/Subtitle Ideas

**Option 1 (Earnings focus):**
```typescript
title: 'Earn with Your GPUs',
subtitle: 'Turn idle hardware into income'
```

**Option 2 (Revenue focus):**
```typescript
title: 'GPU Revenue Calculator',
subtitle: 'See your earning potential'
```

**Option 3 (Opportunity focus):**
```typescript
title: 'Monetize Your Hardware',
subtitle: 'Join the GPU marketplace'
```

## 🚀 Next Step

**[STEP_12_STARTUPS_PAGE.md](./MIGRATION_PLAN_12_STARTUPS_PAGE.md)** - Migrate StartupsPage

---

**Time:** 10 minutes  
**Difficulty:** Easy (image already exists)
