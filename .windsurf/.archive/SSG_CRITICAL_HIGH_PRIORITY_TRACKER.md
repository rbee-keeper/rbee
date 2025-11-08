# SSG Critical + High Priority - Progress Tracker

**Started:** 2025-11-08 01:37 AM  
**Goal:** Fix build-blocking and high-impact serialization issues  
**Estimated Time:** 5 hours

---

## 🎯 Scope: 5 Pages

### 🔴 Critical (Blocking Build)
1. **MultiMachinePage** - 3 JSX props

### 🟠 High Priority (Most JSX)
2. **FeaturesPage** - 20 JSX props
3. **TermsPage** - 18 JSX props (already has asideConfig ✅)
4. **PrivacyPage** - 18 JSX props (already has asideConfig ✅)
5. **PricingPage** - 12 JSX props

**Total JSX Props to Fix:** 71

---

## 📊 Progress Overview

| Page | JSX Props | Status | Time | Notes |
|------|-----------|--------|------|-------|
| MultiMachinePage | 3 | 🔄 IN PROGRESS | 0/1h | Blocking build |
| FeaturesPage | 20 | ⏳ TODO | 0/1.5h | Content props |
| TermsPage | 18 | ⏳ TODO | 0/1h | FAQ answers |
| PrivacyPage | 18 | ⏳ TODO | 0/1h | FAQ answers |
| PricingPage | 12 | ⏳ TODO | 0/1h | FAQ + visual |

**Overall Progress:** 1/5 pages (20%)  
**Time Spent:** 5 min / 5 hours

## ⚠️ New Blocker Discovered

**Page:** RbeeVsOllamaPage (comparison page)  
**Error:** `Functions cannot be passed directly to Client Components`  
**Note:** This page has `'use client'` but still fails SSG prerendering  
**Issue:** Likely in template components, not Props file  
**Action:** Need to investigate template components or add to app route config

---

## 🔴 CRITICAL: MultiMachinePage

**Status:** ✅ COMPLETE  
**Priority:** BLOCKING BUILD  
**JSX Props:** 3

### Issues Found
- [x] Line 27: `subheadline: (JSX)` → Converted to markdown string
- [x] Line 46: `output: (JSX)` → Converted to plain string
- [x] Line 202: `content: (JSX)` → Converted to plain string

### Approach
- ✅ Convert to markdown/plain strings (quick fix)
- ✅ Test build immediately

### Checklist
- [x] Backup Props file
- [x] Fix `subheadline`
- [x] Fix `output`
- [x] Fix `content`
- [x] Test TypeScript compilation
- [x] Test build
- [ ] Verify page renders (build blocked by different page)
- [ ] Commit changes

**Started:** 2025-11-08 01:40 AM  
**Completed:** 2025-11-08 01:45 AM  
**Actual Time:** 5 minutes

**Result:** ✅ MultiMachinePage fixed, but build now blocked by RbeeVsOllamaPage (different issue)

---

## 🟠 HIGH PRIORITY: FeaturesPage

**Status:** ⏳ TODO  
**Priority:** HIGH (20 JSX props)  
**JSX Props:** 20

### Issues Found
- [ ] Multiple `content` props in features
- [ ] Multiple `decoration` props

### Approach
- Create content config system
- Create decoration registry

### Checklist
- [ ] Analyze all JSX props
- [ ] Create config types
- [ ] Update Props file
- [ ] Update Page file (if needed)
- [ ] Test TypeScript compilation
- [ ] Test build
- [ ] Verify page renders
- [ ] Commit changes

**Started:** [TIME]  
**Completed:** [TIME]  
**Actual Time:** [DURATION]

---

## 🟠 HIGH PRIORITY: TermsPage

**Status:** ⏳ TODO  
**Priority:** HIGH (18 JSX props)  
**JSX Props:** 18 (FAQ answers)

### Issues Found
- [ ] 18 FAQ `answer` props with JSX

### Approach
- Convert FAQ answers to markdown
- Keep existing asideConfig ✅

### Checklist
- [ ] Analyze FAQ structure
- [ ] Convert answers to markdown
- [ ] Update FAQ renderer (if needed)
- [ ] Test TypeScript compilation
- [ ] Test build
- [ ] Verify page renders
- [ ] Commit changes

**Started:** [TIME]  
**Completed:** [TIME]  
**Actual Time:** [DURATION]

---

## 🟠 HIGH PRIORITY: PrivacyPage

**Status:** ⏳ TODO  
**Priority:** HIGH (18 JSX props)  
**JSX Props:** 18 (FAQ answers)

### Issues Found
- [ ] 18 FAQ `answer` props with JSX

### Approach
- Convert FAQ answers to markdown
- Keep existing asideConfig ✅

### Checklist
- [ ] Analyze FAQ structure
- [ ] Convert answers to markdown
- [ ] Update FAQ renderer (if needed)
- [ ] Test TypeScript compilation
- [ ] Test build
- [ ] Verify page renders
- [ ] Commit changes

**Started:** [TIME]  
**Completed:** [TIME]  
**Actual Time:** [DURATION]

---

## 🟠 HIGH PRIORITY: PricingPage

**Status:** ⏳ TODO  
**Priority:** HIGH (12 JSX props)  
**JSX Props:** 12

### Issues Found
- [ ] `heading: (JSX)`
- [ ] `visual: (JSX)`
- [ ] Multiple FAQ `answer` props with JSX

### Approach
- Convert heading to markdown/config
- Create visual config
- Convert FAQ answers to markdown

### Checklist
- [ ] Analyze all JSX props
- [ ] Create config types
- [ ] Update Props file
- [ ] Update Page file (if needed)
- [ ] Test TypeScript compilation
- [ ] Test build
- [ ] Verify page renders
- [ ] Commit changes

**Started:** [TIME]  
**Completed:** [TIME]  
**Actual Time:** [DURATION]

---

## 📈 Session Log

### Session 1: [DATE/TIME]
- **Started:** MultiMachinePage
- **Progress:** [DESCRIPTION]
- **Blockers:** [ANY ISSUES]
- **Next:** [NEXT STEPS]

---

## ✅ Completion Criteria

### Per Page
- [ ] No JSX in Props file
- [ ] TypeScript compiles
- [ ] Build succeeds
- [ ] Page renders correctly
- [ ] Changes committed

### Overall
- [ ] Build no longer blocked
- [ ] All 5 pages migrated
- [ ] Documentation updated
- [ ] Patterns documented for future use

---

## 🎯 Success Metrics

### Build Status
```bash
cd frontend/apps/commercial
pnpm build
# Should complete without errors
```

### Pages Fixed
- [ ] 1/5 - MultiMachinePage (critical)
- [ ] 2/5 - FeaturesPage
- [ ] 3/5 - TermsPage
- [ ] 4/5 - PrivacyPage
- [ ] 5/5 - PricingPage

### Time Tracking
- **Estimated:** 5 hours
- **Actual:** [DURATION]
- **Variance:** [DIFFERENCE]

---

**Last Updated:** 2025-11-08 01:50 AM  
**Status:** ⏸️ PAUSED FOR ANALYSIS  
**Current Task:** Root cause analysis complete, instructions ready for next team
