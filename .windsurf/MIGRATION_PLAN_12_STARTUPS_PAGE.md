# Step 12: Migrate StartupsPage

**Phase:** 3 - Low Priority  
**Time:** 10 minutes  
**Priority:** LOW  
**Variant:** ImageAside

## 🎯 Goal

Migrate StartupsPage to use ImageAside component with growth-themed image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/StartupsPage/
├── StartupsPageProps.tsx  → Will rename to .ts
└── StartupsPage.tsx       → Update imports and aside
```

## 🖼️ Image Needed

🔴 **Generate image:** `startups-growth-hero.png` (1536x1024 landscape)

**Theme:** Startup growth, innovation, scaling  
**Aspect ratio:** 3:2 (landscape)

### AI Generation Prompt

```
Create a dynamic startup growth visualization. Show upward trending 
graphs, rocket ship launch, and innovation symbols. Use energetic 
colors with blue (#3b82f6), purple (#8b5cf6), and teal (#06b6d4) 
accents. Include abstract representations of scaling and success. 
Style: Modern startup aesthetic, energetic and optimistic, horizontal 
composition. 1536x1024 pixels, high quality.
```

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) for full details

## ✏️ Changes Needed

### 1. Generate Image First

```bash
cd frontend/apps/commercial/public/images

# Generate with AI tool
# Save as: startups-growth-hero.png (1536x1024)

# Or use placeholder temporarily
# convert -size 1536x1024 xc:lightblue startups-growth-hero.png
```

### 2. Update StartupsPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/startups-growth-hero.png',
  alt: 'Startup growth visualization with upward trending graphs and innovation symbols',
  width: 1536,
  height: 1024,
  title: 'Scale Your Startup',
  subtitle: 'AI infrastructure that grows with you'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 3. Update StartupsPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...startupsHeroProps}
  aside={renderAside(startupsHeroProps.asideConfig)}
/>
```

### 4. Rename Props File

```bash
git mv StartupsPageProps.tsx StartupsPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Generate/add image
# (Use AI tool or temporary placeholder)

# 2. Verify image
test -f public/images/startups-growth-hero.png && echo "✅ Image ready"

# 3. Backup files
cd components/pages/StartupsPage
cp StartupsPageProps.tsx StartupsPageProps.tsx.bak
cp StartupsPage.tsx StartupsPage.tsx.bak

# 4. Edit files

# 5. Test
cd ../../..
pnpm run type-check

# 6. Rename
cd components/pages/StartupsPage
git mv StartupsPageProps.tsx StartupsPageProps.ts

# 7. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image
test -f public/images/startups-growth-hero.png && echo "✅ Image exists"

# Check Props file
grep -q "variant: 'image'" components/pages/StartupsPage/StartupsPageProps.ts && echo "✅ Uses ImageAside"
grep -q "1536x1024" components/pages/StartupsPage/StartupsPageProps.ts && echo "✅ Landscape aspect"

# Check Page file
grep -q "renderAside" components/pages/StartupsPage/StartupsPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/StartupsPage/StartupsPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Image generated/added (1536x1024)
- [ ] Image saved to `/public/images/`
- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with image variant added
- [ ] Image dimensions: 1536x1024 (landscape)
- [ ] Descriptive alt text added
- [ ] Title and subtitle added
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 Landscape Image Details

**Aspect ratio:** 1536x1024 = 3:2 (landscape)  
**Auto class:** `aspect-[3/2]` applied by ImageAside  
**Use for:** Wide content, horizontal layouts

## 🖼️ Image Alternatives

If generation not available:

1. **Use placeholder:**
   ```bash
   convert -size 1536x1024 xc:lightblue public/images/startups-growth-hero.png
   ```

2. **Use IconAside instead:**
   ```typescript
   asideConfig: {
     variant: 'icon',
     icon: 'Rocket',
     title: 'Scale Your Startup',
     subtitle: 'Grow with confidence'
   }
   ```

3. **Use existing image temporarily:**
   ```bash
   # If you have any 1536x1024 image
   cp public/images/some-wide-image.png public/images/startups-growth-hero.png
   ```

## 💡 Title/Subtitle Ideas

**Option 1 (Growth focus):**
```typescript
title: 'Scale Your Startup',
subtitle: 'AI infrastructure that grows with you'
```

**Option 2 (Innovation focus):**
```typescript
title: 'Innovate Faster',
subtitle: 'Focus on product, not infrastructure'
```

**Option 3 (Cost focus):**
```typescript
title: 'Startup-Friendly Pricing',
subtitle: 'Pay only for what you use'
```

## 🚀 Next Step

**[STEP_13_BULK_RENAME.md](./MIGRATION_PLAN_13_BULK_RENAME.md)** - Bulk rename all Props files

---

**Time:** 10 minutes (+ image generation)  
**Difficulty:** Easy (last page migration)
