# Step 8: Migrate HomelabPage

**Phase:** 2 - Medium Priority  
**Time:** 15 minutes  
**Priority:** MEDIUM  
**Variant:** ImageAside

## 🎯 Goal

Migrate HomelabPage to use ImageAside component with existing homelab image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/HomelabPage/
├── HomelabPageProps.tsx  → Will rename to .ts
└── HomelabPage.tsx       → Update imports and aside
```

## 🖼️ Image Available

✅ **Existing image:** `homelab-network.png`

Location: `frontend/apps/commercial/public/images/homelab-network.png`

## ✏️ Changes Needed

### 1. Update HomelabPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/homelab-network.png',
  alt: 'Homelab network setup diagram showing multiple machines connected',
  width: 1024,   // Adjust based on actual image
  height: 1024,  // Adjust based on actual image
  title: 'Multi-Machine Setup',
  subtitle: 'Connect all your hardware'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 2. Update HomelabPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...homelabHeroProps}
  aside={renderAside(homelabHeroProps.asideConfig)}
/>
```

### 3. Rename Props File

```bash
git mv HomelabPageProps.tsx HomelabPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Check image exists and get dimensions
test -f public/images/homelab-network.png && echo "✅ Image exists"
file public/images/homelab-network.png  # Check actual dimensions

# 2. Backup files
cd components/pages/HomelabPage
cp HomelabPageProps.tsx HomelabPageProps.tsx.bak
cp HomelabPage.tsx HomelabPage.tsx.bak

# 3. Edit files
# Note: Adjust width/height based on actual image dimensions

# 4. Test
cd ../../..
pnpm run type-check

# 5. Rename
cd components/pages/HomelabPage
git mv HomelabPageProps.tsx HomelabPageProps.ts

# 6. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image
test -f public/images/homelab-network.png && echo "✅ Image exists"

# Get image info
file public/images/homelab-network.png

# Check Props file
grep -q "variant: 'image'" components/pages/HomelabPage/HomelabPageProps.ts && echo "✅ Uses ImageAside"
grep -q "homelab-network" components/pages/HomelabPage/HomelabPageProps.ts && echo "✅ Correct image"

# Check Page file
grep -q "renderAside" components/pages/HomelabPage/HomelabPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/HomelabPage/HomelabPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Image exists at `/images/homelab-network.png`
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

## 🎨 Image Dimension Guide

Check actual image dimensions and use appropriate aspect ratio:

```bash
# Get image dimensions
file public/images/homelab-network.png
# or
identify public/images/homelab-network.png
```

**Common sizes:**
- 1536x1024 → `width: 1536, height: 1024` (landscape)
- 1024x1024 → `width: 1024, height: 1024` (square)
- 1024x1536 → `width: 1024, height: 1536` (portrait)

ImageAside will automatically apply correct aspect ratio class.

## 💡 Tips

1. **Always check actual dimensions** - Don't guess
2. **Use descriptive alt text** - "Homelab network setup diagram" not "image"
3. **Add context in title/subtitle** - Helps users understand the image
4. **Test rendering** - Verify image loads and looks good

## 🚀 Next Step

**[STEP_09_EDUCATION_PAGE.md](./MIGRATION_PLAN_09_EDUCATION_PAGE.md)** - Migrate EducationPage

---

**Time:** 15 minutes  
**Difficulty:** Easy (image already exists)
