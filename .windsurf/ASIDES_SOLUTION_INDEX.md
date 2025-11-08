# Hero Asides Solution - Complete Index

**TEAM-XXX: CMS-friendly aside components with image support**

## 🎯 What This Solves

**Problem:** JSX elements in Props.tsx cause SSG serialization errors  
**Solution:** Reusable aside components configured from Props.tsx (CMS-friendly)

## 📚 Documentation (Read in Order)

| # | Document | Purpose | Status |
|---|----------|---------|--------|
| 1 | **[ASIDES_SOLUTION_SUMMARY.md](./ASIDES_SOLUTION_SUMMARY.md)** | Overview and benefits | ✅ Complete |
| 2 | **[HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md)** | Complete guide with all variants | ✅ Complete |
| 3 | **[EXAMPLE_TERMS_PAGE_MIGRATION.md](./EXAMPLE_TERMS_PAGE_MIGRATION.md)** | Step-by-step example | ✅ Complete |
| 4 | **[FILE_RENAMING_GUIDE.md](./FILE_RENAMING_GUIDE.md)** | ⚠️ Rename .tsx → .ts | ✅ Complete |
| 5 | **[PLAN.md](./COMMERCIAL_APP_SSG_FIX_PLAN.md)** | Execution plan | ✅ Complete |
| 6 | **[IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md)** | AI prompts for images | ✅ Complete |

## 🚀 Quick Start (3 Steps)

### 1. Read Summary
```bash
cat .windsurf/ASIDES_SOLUTION_SUMMARY.md
```

### 2. See Example
```bash
cat .windsurf/EXAMPLE_TERMS_PAGE_MIGRATION.md
```

### 3. Read File Renaming Guide
```bash
cat .windsurf/FILE_RENAMING_GUIDE.md
```

### 4. Start Migrating
```bash
cat .windsurf/HERO_ASIDES_GUIDE.md
```

## 🎨 4 Aside Variants

| Variant | Use For | Config Example |
|---------|---------|----------------|
| **IconAside** | Legal, callouts | `{ variant: 'icon', icon: 'FileText', title: '...', subtitle: '...' }` |
| **ImageAside** | Visual content | `{ variant: 'image', src: '/images/...', alt: '...', width: 1024, height: 1024 }` |
| **CardAside** | Code examples | `{ variant: 'card', title: '...', icon: 'Code', content: ... }` |
| **StatsAside** | Metrics | `{ variant: 'stats', stats: [...], title: '...' }` |

## 📁 Files Created

### Components
```
frontend/apps/commercial/components/organisms/HeroAsides/
├── HeroAsides.tsx    (4 variants + renderAside)
└── index.ts          (exports)
```

### Documentation
```
.windsurf/
├── ASIDES_SOLUTION_INDEX.md (this file)
├── ASIDES_SOLUTION_SUMMARY.md
├── HERO_ASIDES_GUIDE.md
├── EXAMPLE_TERMS_PAGE_MIGRATION.md
├── COMMERCIAL_APP_SSG_FIX_REVISED_PLAN.md
└── IMAGE_GENERATION_PROMPTS.md
```

## 🎯 The Pattern

### Props.tsx (CMS - Edit Here)
```tsx
import type { AsideConfig } from '../../organisms/HeroAsides'

export const heroProps: HeroTemplateProps = {
  title: "Your Title",
  asideConfig: {
    variant: 'icon',
    icon: 'FileText',
    title: 'Legal Document',
    subtitle: 'Please read carefully'
  } as AsideConfig
}
```

### Page.tsx (Renderer - Just Display)
```tsx
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...heroProps}
  aside={renderAside(heroProps.asideConfig)}
/>
```

## 📋 Pages to Migrate

| Page | Variant | Image? | Priority |
|------|---------|--------|----------|
| TermsPage | IconAside | No | Example |
| PrivacyPage | IconAside | No | HIGH |
| RhaiScriptingPage | ImageAside | ✅ Exists | HIGH |
| DevelopersPage | CardAside | No | HIGH |
| ResearchPage | ImageAside | 🔴 Generate | MEDIUM |
| HomelabPage | ImageAside | ✅ Exists | MEDIUM |
| CommunityPage | StatsAside | Optional | MEDIUM |
| EducationPage | ImageAside | 🔴 Generate | MEDIUM |
| ProvidersPage | ImageAside | ✅ Exists | LOW |
| StartupsPage | ImageAside | 🔴 Generate | LOW |

## 🖼️ Images

### Existing (Ready to Use)
- ✅ `features-rhai-routing.png` (1024x1024) - RhaiScriptingPage
- ✅ `homelab-network.png` - HomelabPage
- ✅ `gpu-earnings.png` - ProvidersPage

### Need to Generate
- 🔴 ResearchPage (1024x1536 portrait)
- 🔴 EducationPage (1024x1536 portrait)
- 🔴 StartupsPage (1536x1024 landscape)
- 🔴 CommunityPage (1024x1024 square) - Optional

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) for AI prompts

## ⏱️ Time Estimate

| Task | Time | Status |
|------|------|--------|
| Components | 30 min | ✅ Done |
| Documentation | 30 min | ✅ Done |
| Page migrations | 1-1.5 hours | 🔴 TODO |
| Image generation | 30-60 min | 🔴 TODO |
| Testing | 15 min | 🔴 TODO |
| **Total** | **2.5-3 hours** | **Ready** |

## ✅ Benefits

| Benefit | Description |
|---------|-------------|
| **CMS-Friendly** | All content in Props.tsx, easy to edit |
| **No Serialization** | Config objects work with SSG |
| **Reusable** | 4 variants cover all cases |
| **Type-Safe** | TypeScript catches errors |
| **Image Support** | Auto aspect ratio handling |
| **Consistent** | Same styling everywhere |

## 🔧 Commands

```bash
# Read all docs
ls -1 .windsurf/ASIDES_*.md .windsurf/HERO_ASIDES_*.md .windsurf/IMAGE_*.md

# View summary
cat .windsurf/ASIDES_SOLUTION_SUMMARY.md

# View guide
cat .windsurf/HERO_ASIDES_GUIDE.md

# View example
cat .windsurf/EXAMPLE_TERMS_PAGE_MIGRATION.md

# View image prompts
cat .windsurf/IMAGE_GENERATION_PROMPTS.md

# Test build
cd frontend/apps/commercial && pnpm build
```

## 📞 FAQ

**Q: Can I still edit content easily?**  
A: YES! All content stays in Props.tsx as before.

**Q: Do I need to know React?**  
A: NO! Just edit the config objects in Props.tsx.

**Q: What if I need a custom aside?**  
A: Add a new variant to HeroAsides.tsx or use CardAside with custom content.

**Q: How do I add images?**  
A: Put in `/public/images/`, reference in `asideConfig.src`.

**Q: Will this break existing pages?**  
A: NO! We're fixing broken pages, not breaking working ones.

## 🎯 Next Steps

1. **Read Summary** - Understand the solution
2. **Read Guide** - Learn all 4 variants
3. **See Example** - TermsPage migration
4. **Generate Images** - Use AI prompts
5. **Migrate Pages** - Follow the pattern
6. **Test Build** - Verify everything works

## 📊 Progress Tracking

- [x] Create reusable components
- [x] Write documentation
- [ ] Migrate TermsPage (example)
- [ ] Migrate remaining 9 pages
- [ ] Generate 3-4 images
- [ ] Test all pages
- [ ] Verify build succeeds

## ✅ Success Criteria

All pages must:
- ✅ Build without errors
- ✅ Render correctly with asides
- ✅ Have content in Props.tsx (CMS)
- ✅ Use reusable components
- ✅ Support SSG (no serialization)

---

**Status:** ✅ Components ready, 🔴 Migrations pending  
**Components:** 4 variants created  
**Documentation:** Complete  
**Next:** Migrate pages, generate images  
**Time:** 2.5-3 hours remaining
