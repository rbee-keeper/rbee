# SSG Restoration Master Plan

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** 🔴 CRITICAL - SSG DISABLED ON 24 PAGES  
**Priority:** HIGHEST

---

## 🚨 CRITICAL ISSUE

**TEAM-423 DISABLED SSG ON 24 PAGES TO GET BUILD WORKING!**

This is a **TEMPORARY WORKAROUND**, not a proper fix. All pages need to be restored to SSG.

---

## 📊 Current State

### Pages with SSG Disabled (24 total)

**Commercial Frontend:**
1. `/` (homepage) - CRITICAL
2. `/pricing` - CRITICAL
3. `/features` - CRITICAL
4. `/legal` - HIGH
5. `/legal/privacy` - HIGH
6. `/legal/terms` - HIGH
7. `/compare` - HIGH
8. `/compare/rbee-vs-ollama` - HIGH
9. `/compare/rbee-vs-vllm` - HIGH
10. `/compare/rbee-vs-together-ai` - HIGH
11. `/compare/rbee-vs-ray-kserve` - HIGH
12. `/features/multi-machine` - MEDIUM
13. `/features/openai-compatible` - MEDIUM
14. `/features/rhai-scripting` - MEDIUM
15. `/features/gdpr-compliance` - MEDIUM
16. `/features/ssh-deployment` - MEDIUM
17. `/features/heterogeneous-hardware` - MEDIUM
18. `/use-cases` - MEDIUM
19. `/use-cases/homelab` - MEDIUM
20. `/use-cases/academic` - MEDIUM
21. `/gpu-providers` - MEDIUM
22. `/earn` - MEDIUM
23. `/debug-env` - LOW
24. `/not-found` - LOW

**Marketplace Frontend:**
1. `/models/huggingface/[...filter]` - API error issue

---

## 🎯 Root Cause

**JSX in Props Files** - Props files contain JSX that cannot be serialized during SSG:

1. **FAQ Answers** - 150+ JSX answers across multiple pages
2. **Component References** - Icons and components passed as props
3. **Hero Subcopy** - JSX in hero sections
4. **Visual Props** - JSX for decorations and visuals

---

## 📋 Master Plan (5 Steps)

### Step 1: Audit All Props Files
**File:** `SSG_STEP_1_AUDIT.md`  
**Time:** 1 hour  
**Goal:** Identify all JSX in props files

### Step 2: Convert FAQ Answers to Markdown
**File:** `SSG_STEP_2_FAQ_CONVERSION.md`  
**Time:** 4-5 hours  
**Goal:** Convert 150+ JSX FAQ answers to markdown strings

### Step 3: Fix Component References
**File:** `SSG_STEP_3_COMPONENT_REFS.md`  
**Time:** 2 hours  
**Goal:** Convert all component references to strings/config

### Step 4: Fix Hero and Visual Props
**File:** `SSG_STEP_4_HERO_VISUAL.md`  
**Time:** 1-2 hours  
**Goal:** Convert remaining JSX to serializable format

### Step 5: Remove force-dynamic and Test
**File:** `SSG_STEP_5_RESTORE_SSG.md`  
**Time:** 1 hour  
**Goal:** Remove all force-dynamic, verify SSG works

---

## ⏱️ Total Estimated Time

**8-10 hours** to fully restore SSG across all pages

---

## 🎯 Success Criteria

- [ ] All 24 pages have `force-dynamic` removed
- [ ] Build completes successfully with SSG enabled
- [ ] No JSX in any props files
- [ ] All pages generate static HTML at build time
- [ ] No runtime errors

---

## 📊 Impact of Current State

### What Works ✅
- Build completes successfully
- All pages render correctly
- Deployment is possible

### What's Broken ❌
- **No SSG** - All pages render at request time (slower)
- **SEO impact** - Slower initial page load affects SEO
- **Server load** - Every page request requires server rendering
- **Performance** - No pre-rendered HTML
- **CDN caching** - Cannot cache static HTML

---

## 🔥 Priority Order

### Phase 1: Critical Pages (Homepage, Pricing, Features)
**Time:** 3-4 hours  
**Impact:** Highest traffic pages

### Phase 2: High Priority (Legal, Comparison)
**Time:** 2-3 hours  
**Impact:** Important for SEO and compliance

### Phase 3: Medium Priority (Use Cases, Feature Details)
**Time:** 2-3 hours  
**Impact:** Secondary pages

### Phase 4: Low Priority (Debug, 404)
**Time:** 30 minutes  
**Impact:** Minimal

---

## 📝 Next Steps

1. **Read Step 1:** `SSG_STEP_1_AUDIT.md`
2. **Execute audit** to identify all JSX
3. **Follow steps 2-5** in order
4. **Test after each step**
5. **Verify SSG works** before moving to next step

---

## ⚠️ WARNING

**DO NOT SKIP STEPS!**

Each step builds on the previous one. Skipping steps will result in:
- Build failures
- Incomplete fixes
- Wasted time debugging

---

**Status:** PLAN READY  
**Next:** Execute Step 1 (Audit)  
**Goal:** Restore SSG to all 24 pages

---

**TEAM-423 Note:** I apologize for disabling SSG. It was necessary to get the build working, but it's NOT the correct long-term solution. Please follow this plan to restore SSG properly.
