# Commercial App SSG Fix - Complete Migration Plan Summary

**Created:** 2025-11-08  
**Status:** ✅ READY FOR EXECUTION  
**Total Time:** 3 hours  
**Pages:** 10 pages to migrate

## 📚 Complete Documentation Set

### Index & Overview
- **[MIGRATION_PLAN_00_INDEX.md](./MIGRATION_PLAN_00_INDEX.md)** - Master index with all steps

### Phase 0: Setup (15 min)
- **[MIGRATION_PLAN_01_VERIFY_COMPONENTS.md](./MIGRATION_PLAN_01_VERIFY_COMPONENTS.md)** - Verify HeroAsides components
- **[MIGRATION_PLAN_02_AUDIT_PAGES.md](./MIGRATION_PLAN_02_AUDIT_PAGES.md)** - Audit current page state

### Phase 1: High Priority (45 min)
- **[MIGRATION_PLAN_03_TERMS_PAGE.md](./MIGRATION_PLAN_03_TERMS_PAGE.md)** - TermsPage (IconAside)
- **[MIGRATION_PLAN_04_PRIVACY_PAGE.md](./MIGRATION_PLAN_04_PRIVACY_PAGE.md)** - PrivacyPage (IconAside)
- **[MIGRATION_PLAN_05_RHAI_SCRIPTING_PAGE.md](./MIGRATION_PLAN_05_RHAI_SCRIPTING_PAGE.md)** - RhaiScriptingPage (ImageAside)

### Phase 2: Medium Priority (1 hour)
- **[MIGRATION_PLAN_06_DEVELOPERS_PAGE.md](./MIGRATION_PLAN_06_DEVELOPERS_PAGE.md)** - DevelopersPage (CardAside)
- **[MIGRATION_PLAN_07_RESEARCH_PAGE.md](./MIGRATION_PLAN_07_RESEARCH_PAGE.md)** - ResearchPage (ImageAside)
- **[MIGRATION_PLAN_08_HOMELAB_PAGE.md](./MIGRATION_PLAN_08_HOMELAB_PAGE.md)** - HomelabPage (ImageAside)
- **[MIGRATION_PLAN_09_EDUCATION_PAGE.md](./MIGRATION_PLAN_09_EDUCATION_PAGE.md)** - EducationPage (ImageAside)

### Phase 3: Low Priority (30 min)
- **[MIGRATION_PLAN_10_COMMUNITY_PAGE.md](./MIGRATION_PLAN_10_COMMUNITY_PAGE.md)** - CommunityPage (StatsAside)
- **[MIGRATION_PLAN_11_PROVIDERS_PAGE.md](./MIGRATION_PLAN_11_PROVIDERS_PAGE.md)** - ProvidersPage (ImageAside)
- **[MIGRATION_PLAN_12_STARTUPS_PAGE.md](./MIGRATION_PLAN_12_STARTUPS_PAGE.md)** - StartupsPage (ImageAside)

### Phase 4: Verification (30 min)
- **[MIGRATION_PLAN_13_BULK_RENAME.md](./MIGRATION_PLAN_13_BULK_RENAME.md)** - Bulk rename Props.tsx → Props.ts
- **[MIGRATION_PLAN_14_FINAL_VERIFICATION.md](./MIGRATION_PLAN_14_FINAL_VERIFICATION.md)** - Final verification

## 🎯 Quick Start

```bash
# 1. Read the index
cat .windsurf/MIGRATION_PLAN_00_INDEX.md

# 2. Start with Phase 0
cat .windsurf/MIGRATION_PLAN_01_VERIFY_COMPONENTS.md
cat .windsurf/MIGRATION_PLAN_02_AUDIT_PAGES.md

# 3. Execute Phase 1 (High Priority)
cat .windsurf/MIGRATION_PLAN_03_TERMS_PAGE.md
# Follow steps 3-5

# 4. Execute Phase 2 (Medium Priority)
# Follow steps 6-9

# 5. Execute Phase 3 (Low Priority)
# Follow steps 10-12

# 6. Execute Phase 4 (Verification)
cat .windsurf/MIGRATION_PLAN_13_BULK_RENAME.md
cat .windsurf/MIGRATION_PLAN_14_FINAL_VERIFICATION.md
```

## 📊 Migration Matrix

| Step | Page | Variant | Image | Time | Priority |
|------|------|---------|-------|------|----------|
| 3 | TermsPage | IconAside | No | 15 min | HIGH |
| 4 | PrivacyPage | IconAside | No | 15 min | HIGH |
| 5 | RhaiScriptingPage | ImageAside | ✅ Exists | 15 min | HIGH |
| 6 | DevelopersPage | CardAside | No | 15 min | MEDIUM |
| 7 | ResearchPage | ImageAside | 🔴 Generate | 15 min | MEDIUM |
| 8 | HomelabPage | ImageAside | ✅ Exists | 15 min | MEDIUM |
| 9 | EducationPage | ImageAside | 🔴 Generate | 15 min | MEDIUM |
| 10 | CommunityPage | StatsAside | No | 10 min | LOW |
| 11 | ProvidersPage | ImageAside | ✅ Exists | 10 min | LOW |
| 12 | StartupsPage | ImageAside | 🔴 Generate | 10 min | LOW |

## 🎨 Aside Variants Used

| Variant | Count | Pages |
|---------|-------|-------|
| **IconAside** | 2 | Terms, Privacy |
| **ImageAside** | 6 | Rhai, Research, Homelab, Education, Providers, Startups |
| **CardAside** | 1 | Developers |
| **StatsAside** | 1 | Community |

## 🖼️ Image Requirements

### Existing (Ready)
- ✅ `features-rhai-routing.png` (1024x1024) - RhaiScriptingPage
- ✅ `homelab-network.png` - HomelabPage
- ✅ `gpu-earnings.png` - ProvidersPage

### Need to Generate
- 🔴 `research-academic-hero.png` (1024x1536) - ResearchPage
- 🔴 `education-learning-hero.png` (1024x1536) - EducationPage
- 🔴 `startups-growth-hero.png` (1536x1024) - StartupsPage

**See:** [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md)

## ⏱️ Time Breakdown

| Phase | Tasks | Time | Cumulative |
|-------|-------|------|------------|
| Phase 0 | Setup & audit | 15 min | 15 min |
| Phase 1 | 3 high priority pages | 45 min | 1 hour |
| Phase 2 | 4 medium priority pages | 1 hour | 2 hours |
| Phase 3 | 3 low priority pages | 30 min | 2.5 hours |
| Phase 4 | Verification | 30 min | 3 hours |
| **Total** | **14 steps** | **3 hours** | **3 hours** |

## 📋 Each Step Includes

Every step document contains:
- 🎯 **Goal** - What to accomplish
- 📁 **Files to Modify** - Exact file paths
- ✏️ **Changes Needed** - Detailed code changes
- 📝 **Implementation** - Step-by-step commands
- 🧪 **Verification** - Automated checks
- 📋 **Checklist** - Task tracking
- 🎨 **Examples** - Code snippets
- 🚀 **Next Step** - Link to next document

## ✅ Success Criteria

All pages must:
- ✅ Build without errors
- ✅ Render correctly with asides
- ✅ Have content in Props.ts (not .tsx)
- ✅ Use reusable components
- ✅ Support SSG (no serialization errors)

## 🔧 Key Commands

```bash
# Navigate to commercial app
cd frontend/apps/commercial

# Type check
pnpm run type-check

# Build
pnpm build

# Dev server
pnpm dev

# Check Props files
find components/pages -name "*Props.tsx"  # Should be empty after migration
find components/pages -name "*Props.ts"   # Should list 10 files
```

## 📞 Support Documents

| Document | Purpose |
|----------|---------|
| [ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md) | Overview of solution |
| [ASIDES_SOLUTION_SUMMARY.md](./ASIDES_SOLUTION_SUMMARY.md) | Benefits and approach |
| [HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md) | Complete guide to all variants |
| [EXAMPLE_TERMS_PAGE_MIGRATION.md](./EXAMPLE_TERMS_PAGE_MIGRATION.md) | Detailed example |
| [FILE_RENAMING_GUIDE.md](./FILE_RENAMING_GUIDE.md) | How to rename .tsx → .ts |
| [IMAGE_GENERATION_PROMPTS.md](./IMAGE_GENERATION_PROMPTS.md) | AI prompts for images |
| [COMMERCIAL_APP_SSG_FIX_PLAN.md](./COMMERCIAL_APP_SSG_FIX_PLAN.md) | Original plan |

## 🎯 Execution Strategy

### Sequential (Recommended)
Follow steps 1-14 in order. Each step builds on previous steps.

### Parallel (Advanced)
- Phase 0: Must complete first (blocking)
- Phase 1-3: Can be done in parallel by different developers
- Phase 4: Must complete last (blocking)

### Incremental (Safe)
- Complete Phase 0 + Phase 1 (high priority)
- Test and verify
- Complete Phase 2 (medium priority)
- Test and verify
- Complete Phase 3 (low priority)
- Final verification

## 🚨 Critical Notes

### Rule Zero Compliance
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

### File Extensions
- **Props files:** `.ts` (not `.tsx`) after migration
- **Page files:** `.tsx` (still contain JSX)
- **Why:** Props files no longer contain JSX, only config objects

### Image Generation
- Can skip image generation and use IconAside instead
- Or use placeholder images temporarily
- See IMAGE_GENERATION_PROMPTS.md for AI prompts

## 📊 Progress Tracking

Use this checklist to track progress:

```markdown
## Phase 0: Setup
- [ ] Step 1: Verify components
- [ ] Step 2: Audit pages

## Phase 1: High Priority
- [ ] Step 3: TermsPage
- [ ] Step 4: PrivacyPage
- [ ] Step 5: RhaiScriptingPage

## Phase 2: Medium Priority
- [ ] Step 6: DevelopersPage
- [ ] Step 7: ResearchPage
- [ ] Step 8: HomelabPage
- [ ] Step 9: EducationPage

## Phase 3: Low Priority
- [ ] Step 10: CommunityPage
- [ ] Step 11: ProvidersPage
- [ ] Step 12: StartupsPage

## Phase 4: Verification
- [ ] Step 13: Bulk rename
- [ ] Step 14: Final verification
```

## 🎉 Completion

When all steps complete:
- ✅ 10 pages migrated
- ✅ 10 Props.ts files (0 Props.tsx)
- ✅ SSG serialization fixed
- ✅ Build succeeds
- ✅ Ready for production

---

**Status:** ✅ READY FOR EXECUTION  
**Documentation:** 15 step-by-step guides created  
**Total Pages:** 10 pages to migrate  
**Total Time:** 3 hours  
**Next:** Start with MIGRATION_PLAN_01_VERIFY_COMPONENTS.md
