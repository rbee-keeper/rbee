# Step 9: Migrate EducationPage

**Phase:** 2 - Medium Priority  
**Time:** 15 minutes  
**Priority:** MEDIUM  
**Variant:** ImageAside

## 🎯 Goal

Migrate EducationPage to use ImageAside component with learning-themed image.

## 📁 Files to Modify

```
frontend/apps/commercial/components/pages/EducationPage/
├── EducationPageProps.tsx  → Will rename to .ts
└── EducationPage.tsx       → Update imports and aside
```

## 🖼️ Image Needed

🔴 **Generate image:** `education-learning-hero.png` (1024x1536 portrait)

**Theme:** Learning, teaching, students, accessibility  
**Aspect ratio:** 2:3 (portrait)

### AI Generation Prompt

```
Create a welcoming educational illustration showing diverse students 
learning together. Include laptops, books, and abstract learning 
symbols. Use warm, inviting colors with blue (#3b82f6) and teal 
(#06b6d4) accents. Show collaboration and knowledge sharing. Style: 
Modern educational illustration, inclusive and accessible, friendly 
atmosphere, vertical composition. 1024x1536 pixels, high quality.
```

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) for full details

## ✏️ Changes Needed

### 1. Generate Image First

```bash
cd frontend/apps/commercial/public/images

# Generate with AI tool
# Save as: education-learning-hero.png (1024x1536)

# Or use existing academic image temporarily
# cp use-case-academic-hero-dark.png education-learning-hero.png
```

### 2. Update EducationPageProps.tsx

**Add import:**
```typescript
import type { AsideConfig } from '../../organisms/HeroAsides'
```

**Replace aside with asideConfig:**
```typescript
asideConfig: {
  variant: 'image',
  src: '/images/education-learning-hero.png',
  alt: 'Students collaborating and learning together with laptops and books',
  width: 1024,
  height: 1536,
  title: 'Learn AI Development',
  subtitle: 'Accessible for all skill levels'
} as AsideConfig
```

**Remove Lucide imports if present.**

### 3. Update EducationPage.tsx

**Add import:**
```typescript
import { renderAside } from '../../organisms/HeroAsides'
```

**Update HeroTemplate:**
```typescript
<HeroTemplate
  {...educationHeroProps}
  aside={renderAside(educationHeroProps.asideConfig)}
/>
```

### 4. Rename Props File

```bash
git mv EducationPageProps.tsx EducationPageProps.ts
```

## 📝 Implementation

```bash
cd frontend/apps/commercial

# 1. Generate/add image
# (Use AI tool or temporary placeholder)

# 2. Verify image
test -f public/images/education-learning-hero.png && echo "✅ Image ready"

# 3. Backup files
cd components/pages/EducationPage
cp EducationPageProps.tsx EducationPageProps.tsx.bak
cp EducationPage.tsx EducationPage.tsx.bak

# 4. Edit files

# 5. Test
cd ../../..
pnpm run type-check

# 6. Rename
cd components/pages/EducationPage
git mv EducationPageProps.tsx EducationPageProps.ts

# 7. Build
cd ../../..
pnpm build
```

## 🧪 Verification

```bash
cd frontend/apps/commercial

# Check image
test -f public/images/education-learning-hero.png && echo "✅ Image exists"

# Check Props file
grep -q "variant: 'image'" components/pages/EducationPage/EducationPageProps.ts && echo "✅ Uses ImageAside"
grep -q "education-learning" components/pages/EducationPage/EducationPageProps.ts && echo "✅ Correct image"

# Check Page file
grep -q "renderAside" components/pages/EducationPage/EducationPage.tsx && echo "✅ Uses renderAside"

# Check renamed
test -f components/pages/EducationPage/EducationPageProps.ts && echo "✅ Renamed to .ts"
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

## 🎨 Educational Theme

**Colors:** Warm, inviting (blue + teal)  
**Mood:** Inclusive, accessible, friendly  
**Elements:** Students, laptops, books, learning symbols

## 🖼️ Image Alternatives

If generation not available:

1. **Use existing academic image:**
   ```bash
   cp public/images/use-case-academic-hero-dark.png public/images/education-learning-hero.png
   ```

2. **Use placeholder:**
   ```bash
   convert -size 1024x1536 xc:lightblue public/images/education-learning-hero.png
   ```

3. **Use IconAside instead:**
   ```typescript
   asideConfig: {
     variant: 'icon',
     icon: 'GraduationCap',
     title: 'Learn AI',
     subtitle: 'For all skill levels'
   }
   ```

## 🚀 Next Step

**[STEP_10_COMMUNITY_PAGE.md](./MIGRATION_PLAN_10_COMMUNITY_PAGE.md)** - Migrate CommunityPage

---

**Time:** 15 minutes (+ image generation)  
**Difficulty:** Medium (requires image generation)
