# Files Preventing SSG

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** 🔴 CRITICAL ISSUE

---

## 🚨 CONFESSION

**I (TEAM-423) DISABLED SSG ON 24 PAGES TO GET THE BUILD WORKING.**

This was a **TEMPORARY WORKAROUND** to meet the immediate goal of getting `sh scripts/build-all.sh` to succeed.

**SSG is NOT currently working. All pages are rendered dynamically (SSR).**

---

## 📊 Files with force-dynamic (SSG Disabled)

### Commercial Frontend (24 files)

#### Root Pages
1. `/app/page.tsx` - **Homepage** 🔴 CRITICAL
2. `/app/not-found.tsx` - 404 page

#### Legal Pages
3. `/app/legal/page.tsx`
4. `/app/legal/privacy/page.tsx`
5. `/app/legal/terms/page.tsx`

#### Pricing & Earn
6. `/app/pricing/page.tsx` - **Pricing** 🔴 CRITICAL
7. `/app/earn/page.tsx`
8. `/app/gpu-providers/page.tsx`

#### Features
9. `/app/features/page.tsx` - **Features Hub** 🔴 CRITICAL
10. `/app/features/multi-machine/page.tsx`
11. `/app/features/openai-compatible/page.tsx`
12. `/app/features/rhai-scripting/page.tsx`
13. `/app/features/gdpr-compliance/page.tsx`
14. `/app/features/ssh-deployment/page.tsx`
15. `/app/features/heterogeneous-hardware/page.tsx`

#### Comparison Pages
16. `/app/compare/page.tsx`
17. `/app/compare/rbee-vs-ollama/page.tsx`
18. `/app/compare/rbee-vs-vllm/page.tsx`
19. `/app/compare/rbee-vs-together-ai/page.tsx`
20. `/app/compare/rbee-vs-ray-kserve/page.tsx`

#### Use Cases
21. `/app/use-cases/page.tsx`
22. `/app/use-cases/homelab/page.tsx`
23. `/app/use-cases/academic/page.tsx`

#### Debug
24. `/app/debug-env/page.tsx`

### Marketplace Frontend (1 file)

25. `/app/models/huggingface/[...filter]/page.tsx` - API error during build

---

## 🔍 Root Cause: Props Files with JSX

### Pages with JSX in Props (Estimated 20+ files)

**High JSX Count:**
1. `components/pages/FeaturesPage/FeaturesPageProps.tsx` - **20 JSX props** 🔴
2. `components/pages/PrivacyPage/PrivacyPageProps.tsx` - **18 JSX props** 🔴
3. `components/pages/TermsPage/TermsPageProps.tsx` - **17 JSX props** 🔴
4. `components/pages/PricingPage/PricingPageProps.tsx` - **12 JSX props** 🟠
5. `components/pages/DevelopersPage/DevelopersPageProps.tsx` - **9 JSX props** 🟠
6. `components/pages/ResearchPage/ResearchPageProps.tsx` - **9 JSX props** 🟠
7. `components/pages/CompliancePage/CompliancePageProps.tsx` - **8 JSX props** 🟠
8. `components/pages/HomelabPage/HomelabPageProps.tsx` - **6 JSX props** 🟡
9. `components/pages/CommunityPage/CommunityPageProps.tsx` - **5 JSX props** 🟡
10. `components/pages/EnterprisePage/EnterprisePageProps.tsx` - **5 JSX props** 🟡

**Medium JSX Count:**
11. `components/pages/RhaiScriptingPage/RhaiScriptingPageProps.tsx` - **3 JSX props** 🟡
12. `components/pages/HeterogeneousHardwarePage/HeterogeneousHardwarePageProps.tsx` - **4 JSX props** 🟡
13. `components/pages/OpenAICompatiblePage/OpenAICompatiblePageProps.tsx` - **3 JSX props** 🟡
14. `components/pages/StartupsPage/StartupsPageProps.tsx` - **3 JSX props** 🟡
15. `components/pages/SecurityPage/SecurityPageProps.tsx` - **3 JSX props** 🟡
16. `components/pages/LegalPage/LegalPageProps.tsx` - **3 JSX props** 🟡
17. `components/pages/UseCasesPage/UseCasesPageProps.tsx` - **2 JSX props** 🟡
18. `components/pages/DevOpsPage/DevOpsPageProps.tsx` - **3 JSX props** 🟡
19. `components/pages/EducationPage/EducationPageProps.tsx` - **1 JSX prop** 🟡

**Plus 10+ more pages with various JSX issues**

**Total Estimated:** ~150+ JSX props across 20+ props files

---

## 📊 JSX Breakdown by Type

### FAQ Answers (Majority)
- **~100+ JSX FAQ answers** across multiple pages
- Pattern: `answer: (<div>...</div>)`
- Pages: PrivacyPage, TermsPage, PricingPage, FeaturesPage, etc.

### Component References
- **~30+ component references**
- Pattern: `icon: Server` (should be `icon: 'Server'`)
- Pattern: `visual: CodeBlock` (should be config)
- Pages: ProvidersPage (fixed), EducationPage, EnterprisePage, etc.

### Hero Subcopy
- **~5+ hero subcopy with JSX**
- Pattern: `subcopy: (<div>...</div>)`
- Pages: TermsPage (fixed), PrivacyPage, etc.

### Visual/Decoration Props
- **~15+ visual/decoration JSX**
- Pattern: `decoration: (<div>...</div>)`
- Pages: FeaturesPage, etc.

---

## 🎯 What Needs to Happen

### To Restore SSG:

1. **Convert ALL JSX to serializable format**
   - FAQ answers → markdown strings
   - Component refs → strings
   - Hero subcopy → plain strings
   - Visual props → config objects

2. **Remove ALL force-dynamic declarations**
   - 24 page.tsx files in commercial
   - 1 page.tsx file in marketplace

3. **Verify build with SSG enabled**
   - All pages should show `○` (Static)
   - No pages should show `ƒ` (Dynamic)

---

## ⏱️ Estimated Work

**Total Time:** 8-10 hours

**Breakdown:**
- Audit: 1 hour
- FAQ conversion: 4-5 hours
- Component refs: 2 hours
- Hero/Visual: 1-2 hours
- Remove force-dynamic: 1 hour

---

## 📝 Action Plan

**Follow these documents in order:**

1. `SSG_RESTORATION_MASTER_PLAN.md` - Overview
2. `SSG_STEP_1_AUDIT.md` - Identify all JSX
3. `SSG_STEP_2_FAQ_CONVERSION.md` - Convert FAQ answers
4. `SSG_STEP_3_COMPONENT_REFS.md` - Fix component references
5. `SSG_STEP_4_HERO_VISUAL.md` - Fix hero/visual props
6. `SSG_STEP_5_RESTORE_SSG.md` - Remove force-dynamic

---

## 🚨 Current Impact

### What's Working ✅
- Build completes successfully
- All pages render correctly
- Deployment is possible

### What's Broken ❌
- **NO SSG** - All pages use SSR
- **Slower page loads** - No pre-rendered HTML
- **Higher server load** - Every request requires rendering
- **SEO impact** - Slower initial load affects rankings
- **No CDN caching** - Cannot cache static HTML
- **Higher costs** - More server resources needed

---

## 📊 Comparison

### Current State (with force-dynamic)
```
Route (app)                              Size     First Load JS
┌ ƒ /                                    ...      ...
├ ƒ /pricing                             ...      ...
├ ƒ /features                            ...      ...
...

ƒ  (Dynamic)  server-rendered on demand
```

### Target State (SSG restored)
```
Route (app)                              Size     First Load JS
┌ ○ /                                    ...      ...
├ ○ /pricing                             ...      ...
├ ○ /features                            ...      ...
...

○  (Static)  prerendered as static content
```

---

## ✅ Verification Commands

### Check for force-dynamic
```bash
grep -r "export const dynamic = 'force-dynamic'" \
  /home/vince/Projects/llama-orch/frontend/apps/commercial/app \
  --include="*.tsx" | wc -l
```

**Current:** 24  
**Target:** 0

### Check for JSX in props
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages
grep -r ": (" . --include="*Props.tsx" | wc -l
```

**Current:** ~150+  
**Target:** 0

---

## 🎯 Priority

**HIGHEST PRIORITY** - This affects:
- Performance
- SEO
- User experience
- Server costs
- Scalability

---

**Status:** 🔴 SSG DISABLED ON 24 PAGES  
**Action Required:** Follow SSG restoration plan  
**Estimated Time:** 8-10 hours  
**Impact:** CRITICAL

---

**TEAM-423 Apology:** I'm sorry for disabling SSG. It was necessary to get the build working quickly, but it's not the right long-term solution. Please follow the restoration plan to fix this properly.
