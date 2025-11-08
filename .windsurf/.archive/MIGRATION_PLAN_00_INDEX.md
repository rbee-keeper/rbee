# Commercial App SSG Fix - Migration Plan Index

**TEAM-XXX: Complete step-by-step migration plan**

## 🎯 Mission

Fix SSG serialization errors by migrating 10 pages to use reusable aside components configured from Props files.

## 📋 Overview

| Phase | Steps | Time | Status |
|-------|-------|------|--------|
| **Phase 0: Setup** | Steps 1-2 | 15 min | 🔴 TODO |
| **Phase 1: High Priority** | Steps 3-5 | 45 min | 🔴 TODO |
| **Phase 2: Medium Priority** | Steps 6-9 | 1 hour | 🔴 TODO |
| **Phase 3: Low Priority** | Steps 10-12 | 30 min | 🔴 TODO |
| **Phase 4: Verification** | Steps 13-14 | 30 min | 🔴 TODO |
| **Total** | 14 steps | **3 hours** | **Ready** |

## 📚 Step Documents

### Phase 0: Setup & Verification
- **[STEP_01_VERIFY_COMPONENTS.md](./MIGRATION_PLAN_01_VERIFY_COMPONENTS.md)** - Verify HeroAsides components exist
- **[STEP_02_AUDIT_PAGES.md](./MIGRATION_PLAN_02_AUDIT_PAGES.md)** - Audit current page state

### Phase 1: High Priority Pages (Legal + Critical Features)
- **[STEP_03_TERMS_PAGE.md](./MIGRATION_PLAN_03_TERMS_PAGE.md)** - TermsPage (IconAside)
- **[STEP_04_PRIVACY_PAGE.md](./MIGRATION_PLAN_04_PRIVACY_PAGE.md)** - PrivacyPage (IconAside)
- **[STEP_05_RHAI_SCRIPTING_PAGE.md](./MIGRATION_PLAN_05_RHAI_SCRIPTING_PAGE.md)** - RhaiScriptingPage (ImageAside)

### Phase 2: Medium Priority Pages (Developer + Academic)
- **[STEP_06_DEVELOPERS_PAGE.md](./MIGRATION_PLAN_06_DEVELOPERS_PAGE.md)** - DevelopersPage (CardAside)
- **[STEP_07_RESEARCH_PAGE.md](./MIGRATION_PLAN_07_RESEARCH_PAGE.md)** - ResearchPage (ImageAside)
- **[STEP_08_HOMELAB_PAGE.md](./MIGRATION_PLAN_08_HOMELAB_PAGE.md)** - HomelabPage (ImageAside)
- **[STEP_09_EDUCATION_PAGE.md](./MIGRATION_PLAN_09_EDUCATION_PAGE.md)** - EducationPage (ImageAside)

### Phase 3: Low Priority Pages (Business + Community)
- **[STEP_10_COMMUNITY_PAGE.md](./MIGRATION_PLAN_10_COMMUNITY_PAGE.md)** - CommunityPage (StatsAside)
- **[STEP_11_PROVIDERS_PAGE.md](./MIGRATION_PLAN_11_PROVIDERS_PAGE.md)** - ProvidersPage (ImageAside)
- **[STEP_12_STARTUPS_PAGE.md](./MIGRATION_PLAN_12_STARTUPS_PAGE.md)** - StartupsPage (ImageAside)

### Phase 4: Verification & Cleanup
- **[STEP_13_BULK_RENAME.md](./MIGRATION_PLAN_13_BULK_RENAME.md)** - Bulk rename Props.tsx → Props.ts
- **[STEP_14_FINAL_VERIFICATION.md](./MIGRATION_PLAN_14_FINAL_VERIFICATION.md)** - Build, test, verify

## 🎨 Aside Variants Summary

| Variant | Pages Using | Config Example |
|---------|-------------|----------------|
| **IconAside** | Terms, Privacy | `{ variant: 'icon', icon: 'FileText', title: '...', subtitle: '...' }` |
| **ImageAside** | Rhai, Research, Homelab, Education, Providers, Startups | `{ variant: 'image', src: '/images/...', alt: '...', width: 1024, height: 1024 }` |
| **CardAside** | Developers | `{ variant: 'card', title: '...', icon: 'Code', content: ... }` |
| **StatsAside** | Community | `{ variant: 'stats', stats: [...], title: '...' }` |

## 🖼️ Image Requirements

### Existing Images (Ready)
- ✅ `features-rhai-routing.png` (1024x1024) - RhaiScriptingPage
- ✅ `homelab-network.png` - HomelabPage
- ✅ `gpu-earnings.png` - ProvidersPage

### Images to Generate
- 🔴 ResearchPage - Academic theme (1024x1536 portrait)
- 🔴 EducationPage - Learning theme (1024x1536 portrait)
- 🔴 StartupsPage - Growth theme (1536x1024 landscape)
- 🔴 CommunityPage - Optional (can use stats instead)

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md)

## ⏱️ Time Breakdown

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 0 | Setup verification | 15 min |
| Phase 1 | 3 high priority pages | 45 min (15 min each) |
| Phase 2 | 4 medium priority pages | 1 hour (15 min each) |
| Phase 3 | 3 low priority pages | 30 min (10 min each) |
| Phase 4 | Verification | 30 min |
| **Total** | **14 steps** | **3 hours** |

## 🎯 Success Criteria

Each page must:
- ✅ Build without errors
- ✅ Render correctly with aside
- ✅ Have content in Props.ts (not .tsx)
- ✅ Use reusable components
- ✅ Support SSG (no serialization errors)

## 🚀 Quick Start

```bash
# 1. Read this index
cat .windsurf/MIGRATION_PLAN_00_INDEX.md

# 2. Start with Step 1
cat .windsurf/MIGRATION_PLAN_01_VERIFY_COMPONENTS.md

# 3. Follow steps in order
# Each step is self-contained with:
# - Current state analysis
# - Changes needed
# - Code examples
# - Verification commands
```

## 📊 Progress Tracking

### Phase 0: Setup
- [ ] Step 1: Verify components exist
- [ ] Step 2: Audit current pages

### Phase 1: High Priority
- [ ] Step 3: TermsPage (IconAside)
- [ ] Step 4: PrivacyPage (IconAside)
- [ ] Step 5: RhaiScriptingPage (ImageAside)

### Phase 2: Medium Priority
- [ ] Step 6: DevelopersPage (CardAside)
- [ ] Step 7: ResearchPage (ImageAside)
- [ ] Step 8: HomelabPage (ImageAside)
- [ ] Step 9: EducationPage (ImageAside)

### Phase 3: Low Priority
- [ ] Step 10: CommunityPage (StatsAside)
- [ ] Step 11: ProvidersPage (ImageAside)
- [ ] Step 12: StartupsPage (ImageAside)

### Phase 4: Verification
- [ ] Step 13: Bulk rename .tsx → .ts
- [ ] Step 14: Final verification

## 🔧 Commands Reference

```bash
# Navigate to commercial app
cd frontend/apps/commercial

# Type check
pnpm run type-check

# Build
pnpm build

# Dev server
pnpm dev

# Check for remaining .tsx Props files
find components/pages -name "*Props.tsx" -not -name "*.bak"

# Check for .ts Props files
find components/pages -name "*Props.ts"
```

## 📞 Support Documents

| Document | Purpose |
|----------|---------|
| [ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md) | Overview of solution |
| [HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md) | Complete guide to all variants |
| [EXAMPLE_TERMS_PAGE_MIGRATION.md](./EXAMPLE_TERMS_PAGE_MIGRATION.md) | Detailed example |
| [FILE_RENAMING_GUIDE.md](./FILE_RENAMING_GUIDE.md) | How to rename .tsx → .ts |
| [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) | AI prompts for images |

## ✅ Rule Zero Compliance

✅ **What we're doing RIGHT:**
- Creating reusable components (not one-off fixes)
- Updating architecture (proper separation)
- No backwards compatibility wrappers
- Deleting problematic patterns

❌ **What we're NOT doing:**
- Creating `renderAside_v2()` functions
- Keeping old JSX patterns "for compatibility"
- Adding temporary workarounds
- Creating helper scripts

---

**Status:** Ready for execution  
**Total Time:** 3 hours  
**Pages:** 10 pages to migrate  
**Components:** ✅ Ready (4 variants)  
**Documentation:** ✅ Complete (14 step guides)
