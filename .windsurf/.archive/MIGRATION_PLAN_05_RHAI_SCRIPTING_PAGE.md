# Step 5: Migrate RhaiScriptingPage

**Phase:** 1 - High Priority  
**Time:** 15 minutes  
**Priority:** HIGH  
**Variant:** ImageAside

## 🎯 Goal

Migrate RhaiScriptingPage to use ImageAside component with existing image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/RhaiScriptingPage/
├── RhaiScriptingPageProps.tsx  → Will rename to .ts
└── RhaiScriptingPage.tsx       → Update imports and aside
```

## 🖼️ Image Available

✅ **Existing image:** `features-rhai-routing.png` (1024x1024)

Location: `frontend/apps/commercial/public/images/features-rhai-routing.png`

## ✏️ Changes Needed

### 1. Update RhaiScriptingPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/features-rhai-routing.png',
  alt: 'Rhai scripting interface showing code flow and routing logic',
  width: 1024,
  height: 1024,
  title: 'User-Scriptable Routing',
  subtitle: 'Write custom logic in Rhai'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update RhaiScriptingPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...rhaiScriptingHeroProps}
  aside={renderAside(rhaiScriptingHeroProps.asideConfig)}
/>
```

**Remove Lucide imports if present.**

### 3. Rename Props File

```bash
git mv RhaiScriptingPageProps.tsx RhaiScriptingPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Verify image exists
test -f public/images/features-rhai-routing.png && echo "✅ Image exists" || echo "❌ Image missing"

# 2. Backup files
cd components/pages/RhaiScriptingPage
cp RhaiScriptingPageProps.tsx RhaiScriptingPageProps.tsx.bak
cp RhaiScriptingPage.tsx RhaiScriptingPage.tsx.bak

# 3. Edit files

# 4. Test
cd ../../..
pnpm run type-check

# 5. Rename
cd components/pages/RhaiScriptingPage
git mv RhaiScriptingPageProps.tsx RhaiScriptingPageProps.ts

# 6. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image exists
test -f public/images/features-rhai-routing.png && echo "✅ Image exists"

# Check Props file
! grep -q "<" components/pages/RhaiScriptingPage/RhaiScriptingPageProps.ts && echo "✅ No JSX"
grep -q "asideConfig" components/pages/RhaiScriptingPage/RhaiScriptingPageProps.ts && echo "✅ Has asideConfig"
grep -q "variant: 'image'" components/pages/RhaiScriptingPage/RhaiScriptingPageProps.ts && echo "✅ Uses ImageAside"

# Check Page file
grep -q "renderAside" components/pages/RhaiScriptingPage/RhaiScriptingPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/RhaiScriptingPage/RhaiScriptingPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Image exists at `/images/features-rhai-routing.png`
- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with image variant added
- [ ] Image src, alt, dimensions specified
- [ ] Title and subtitle added
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 ImageAside Config Details

```typescript
asideConfig: {
  variant: 'image',
  src: '/images/features-rhai-routing.png',  // Path from public/
  alt: 'Rhai scripting interface showing code flow and routing logic',  // Descriptive alt text
  width: 1024,   // Image width
  height: 1024,  // Image height (1:1 aspect ratio)
  title: 'User-Scriptable Routing',  // Optional caption title
  subtitle: 'Write custom logic in Rhai'  // Optional caption subtitle
}
```

**Aspect ratio:** 1024x1024 = 1:1 (square) → `aspect-square` class applied automatically

## 🖼️ Image Best Practices

- ✅ Use descriptive alt text (accessibility)
- ✅ Specify exact dimensions (prevents layout shift)
- ✅ Use Next.js Image component (automatic optimization)
- ✅ Add title/subtitle for context

## 🚀 Next Step

**[STEP_06_DEVELOPERS_PAGE.md](./MIGRATION_PLAN_06_DEVELOPERS_PAGE.md)** - Migrate DevelopersPage

---

**Time:** 15 minutes  
**Difficulty:** Easy (image already exists)
