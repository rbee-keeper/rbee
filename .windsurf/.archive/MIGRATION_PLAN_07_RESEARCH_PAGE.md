# Step 7: Migrate ResearchPage

**Phase:** 2 - Medium Priority  
**Time:** 15 minutes  
**Priority:** MEDIUM  
**Variant:** ImageAside

## 🎯 Goal

Migrate ResearchPage to use ImageAside component with academic-themed image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/ResearchPage/
├── ResearchPageProps.tsx  → Will rename to .ts
└── ResearchPage.tsx       → Update imports and aside
```

## 🖼️ Image Needed

🔴 **Generate image:** `research-academic-hero.png` (1024x1536 portrait)

**Theme:** Academic research, AI/ML, scholarly  
**Aspect ratio:** 2:3 (portrait)

### AI Generation Prompt

```
Create an elegant academic research illustration. Show neural network 
diagrams, research papers, and data visualizations in a scholarly 
setting. Use a light, professional color palette with blue (#3b82f6) 
and purple (#8b5cf6) accents. Include abstract representations of AI 
models and research graphs. Style: Modern academic illustration, clean 
and trustworthy, vertical composition. 1024x1536 pixels, high quality.
```

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) for full details

## ✏️ Changes Needed

### 1. Generate Image First

```bash
cd frontend/apps/commercial/public/images

# Option 1: Generate with AI (DALL-E, Midjourney, etc.)
# Save as: research-academic-hero.png (1024x1536)

# Option 2: Use existing image temporarily
# cp og-academic.png research-academic-hero.png

# Verify
ls -lh research-academic-hero.png
# Should be 1024x1536 PNG
```

### 2. Update ResearchPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/research-academic-hero.png',
  alt: 'Academic research visualization with neural networks and data graphs',
  width: 1024,
  height: 1536,
  title: 'Research-Grade Infrastructure',
  subtitle: 'Built for academic excellence'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 3. Update ResearchPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...researchHeroProps}
  aside={renderAside(researchHeroProps.asideConfig)}
/>
```

### 4. Rename Props File

```bash
git mv ResearchPageProps.tsx ResearchPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Generate/add image
# (Use AI tool or temporary placeholder)

# 2. Verify image
test -f public/images/research-academic-hero.png && echo "✅ Image ready" || echo "❌ Generate image first"

# 3. Backup files
cd components/pages/ResearchPage
cp ResearchPageProps.tsx ResearchPageProps.tsx.bak
cp ResearchPage.tsx ResearchPage.tsx.bak

# 4. Edit files

# 5. Test
cd ../../..
pnpm run type-check

# 6. Rename
cd components/pages/ResearchPage
git mv ResearchPageProps.tsx ResearchPageProps.ts

# 7. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image
test -f public/images/research-academic-hero.png && echo "✅ Image exists"
file public/images/research-academic-hero.png | grep -q "1024 x 1536" && echo "✅ Correct size"

# Check Props file
grep -q "variant: 'image'" components/pages/ResearchPage/ResearchPageProps.ts && echo "✅ Uses ImageAside"
grep -q "1024x1536" components/pages/ResearchPage/ResearchPageProps.ts && echo "✅ Portrait aspect"

# Check Page file
grep -q "renderAside" components/pages/ResearchPage/ResearchPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/ResearchPage/ResearchPageProps.ts && echo "✅ Renamed to .ts"
```

## 📋 Checklist

- [ ] Image generated/added (1024x1536)
- [ ] Image saved to `/public/images/`
- [ ] Files backed up
- [ ] AsideConfig import added
- [ ] asideConfig with image variant added
- [ ] Image dimensions: 1024x1536 (portrait)
- [ ] Descriptive alt text added
- [ ] Title and subtitle added
- [ ] Lucide imports removed from Props
- [ ] renderAside import added to Page
- [ ] aside prop updated in Page
- [ ] Type check passes
- [ ] Renamed to .ts
- [ ] Build succeeds

## 🎨 Portrait Image Details

**Aspect ratio:** 1024x1536 = 2:3 (portrait)  
**Auto class:** `aspect-[2/3]` applied by ImageAside  
**Use for:** Tall content, vertical layouts

## 🖼️ Image Alternatives

If generation not available:

1. **Use existing image temporarily:**
   ```bash
   cp public/images/og-academic.png public/images/research-academic-hero.png
   ```

2. **Use placeholder:**
   ```bash
   # Create 1024x1536 placeholder
   convert -size 1024x1536 xc:lightblue public/images/research-academic-hero.png
   ```

3. **Skip image, use IconAside instead:**
   ```typescript
   asideConfig: {
     variant: 'icon',
     icon: 'GraduationCap',
     title: 'Research-Grade',
     subtitle: 'Academic excellence'
   }
   ```

## 🚀 Next Step

**[STEP_08_HOMELAB_PAGE.md](./MIGRATION_PLAN_08_HOMELAB_PAGE.md)** - Migrate HomelabPage

---

**Time:** 15 minutes (+ image generation)  
**Difficulty:** Medium (requires image generation)
