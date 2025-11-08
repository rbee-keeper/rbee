# Full SSG Enablement Plan

**Date:** 2025-11-08  
**Status:** 📋 PLAN  
**Goal:** Enable Static Site Generation for all commercial app pages

---

## 🎯 Executive Summary

**Current Status:**
- ✅ 7 pages have SSG-compatible asides
- ❌ 21+ pages have JSX serialization issues
- ❌ Build fails on `/features/multi-machine`

**Goal:** Make all pages SSG-compatible by eliminating JSX in Props files

---

## 📊 Serialization Issues Found

### Critical (Blocking Build) - 1 Page

| Page | JSX Props | Issue Type | Priority |
|------|-----------|------------|----------|
| **MultiMachinePage** | 3 | `subheadline`, `output`, `content` | 🔴 CRITICAL |

**This page is blocking the build!**

### High Priority - 9 Pages

| Page | JSX Props | Main Issues |
|------|-----------|-------------|
| **FeaturesPage** | 20 | Multiple `content` props, decorations |
| **TermsPage** | 18 | FAQ `answer` props (already has asideConfig ✅) |
| **PrivacyPage** | 18 | FAQ `answer` props (already has asideConfig ✅) |
| **PricingPage** | 12 | `heading`, `visual`, FAQ answers |
| **ProvidersPage** | 10 | Multiple `content` props |
| **ResearchPage** | 9 | `decoration`, `content` (already has asideConfig ✅) |
| **DevelopersPage** | 9 | `subheadline`, `output`, terminal lines |
| **CompliancePage** | 8 | Multiple `decoration` props |
| **HomelabPage** | 6 | `aside`, terminal content (already has asideConfig ✅) |

### Medium Priority - 7 Pages

| Page | JSX Props |
|------|-----------|
| **EnterprisePage** | 5 |
| **CommunityPage** | 5 (already has asideConfig ✅) |
| **HeterogeneousHardwarePage** | 4 |
| **StartupsPage** | 3 (already has asideConfig ✅) |
| **SecurityPage** | 3 |
| **RhaiScriptingPage** | 3 |
| **OpenAICompatiblePage** | 3 |

### Low Priority - 4 Pages

| Page | JSX Props |
|------|-----------|
| **LegalPage** | 3 |
| **DevOpsPage** | 3 |
| **UseCasesPage** | 2 |
| **EducationPage** | 1 (already has asideConfig ✅) |

**Total:** 21 pages with serialization issues

---

## 🔍 Issue Categories

### 1. FAQ Answers (Most Common)
**Pages:** TermsPage, PrivacyPage, PricingPage, etc.

**Problem:**
```typescript
answer: (
  <div className="prose">
    <p>Content here...</p>
  </div>
)
```

**Solution:**
```typescript
answer: "Content here..." // Use markdown string
// OR
answerConfig: {
  type: 'prose',
  content: 'Content here...'
}
```

### 2. Subheadlines / Content
**Pages:** MultiMachinePage, DevelopersPage, FeaturesPage

**Problem:**
```typescript
subheadline: (
  <>Text with <strong>bold</strong></>
)
```

**Solution:**
```typescript
subheadline: "Text with **bold**" // Use markdown
// OR
subheadlineConfig: {
  text: "Text with",
  highlight: "bold"
}
```

### 3. Terminal Output
**Pages:** DevelopersPage, HomelabPage, MultiMachinePage

**Problem:**
```typescript
output: (
  <div className="space-y-1">
    <div className="text-chart-2">Output</div>
  </div>
)
```

**Solution:**
```typescript
outputConfig: {
  lines: [
    { text: "Output", color: "chart-2" }
  ]
}
```

### 4. Decorations
**Pages:** ResearchPage, CompliancePage, FeaturesPage

**Problem:**
```typescript
decoration: (
  <div className="...">
    <NetworkMesh />
  </div>
)
```

**Solution:**
```typescript
decorationConfig: {
  type: 'network-mesh',
  className: '...'
}
```

### 5. Visual Elements
**Pages:** PricingPage, ProvidersPage

**Problem:**
```typescript
visual: (
  <Card>...</Card>
)
```

**Solution:**
```typescript
visualConfig: {
  type: 'card',
  // ... config
}
```

---

## 📋 Implementation Plan

### Phase 1: Critical (Unblock Build) - 2 hours

**Goal:** Fix MultiMachinePage to enable builds

1. **MultiMachinePage** (30 min)
   - Create config types for `subheadline`, `output`, `content`
   - Update Props file
   - Update Page file
   - Test build

2. **Verify Build** (30 min)
   - Run full build
   - Identify next blocking issue
   - Document any new errors

### Phase 2: High Priority Pages - 8 hours

**Goal:** Fix pages with most JSX props

#### Group A: FAQ-Heavy Pages (3 hours)
1. **TermsPage** - Convert FAQ answers to markdown
2. **PrivacyPage** - Convert FAQ answers to markdown
3. **PricingPage** - Convert FAQ answers + heading/visual

#### Group B: Content-Heavy Pages (3 hours)
4. **FeaturesPage** - Create content config system
5. **ProvidersPage** - Convert content props
6. **ResearchPage** - Convert decoration + content

#### Group C: Developer Pages (2 hours)
7. **DevelopersPage** - Convert subheadline, output, terminal
8. **CompliancePage** - Convert decorations
9. **HomelabPage** - Convert terminal content

### Phase 3: Medium Priority Pages - 4 hours

10. **EnterprisePage**
11. **CommunityPage** (already has asideConfig)
12. **HeterogeneousHardwarePage**
13. **StartupsPage** (already has asideConfig)
14. **SecurityPage**
15. **RhaiScriptingPage**
16. **OpenAICompatiblePage**

### Phase 4: Low Priority Pages - 2 hours

17. **LegalPage**
18. **DevOpsPage**
19. **UseCasesPage**
20. **EducationPage** (already has asideConfig)

### Phase 5: Verification - 2 hours

- Run full build
- Test all pages
- Verify SSG output
- Performance testing

**Total Estimated Time:** 18 hours

---

## 🛠️ Technical Approach

### Strategy 1: Markdown Conversion (Recommended for FAQ)

**Best for:** FAQ answers, simple content

```typescript
// Before
answer: (<div className="prose"><p>Text</p></div>)

// After
answer: "Text" // Render with markdown processor
```

**Pros:**
- ✅ Simple
- ✅ CMS-friendly
- ✅ Serializable

**Cons:**
- ⚠️ Limited styling control

### Strategy 2: Config Objects (Recommended for Complex Content)

**Best for:** Terminal output, decorations, visual elements

```typescript
// Before
output: (
  <div className="space-y-1">
    <div className="text-chart-2">$ command</div>
    <div className="text-chart-3">→ Output</div>
  </div>
)

// After
outputConfig: {
  lines: [
    { text: "$ command", color: "chart-2" },
    { text: "→ Output", color: "chart-3" }
  ]
}
```

**Pros:**
- ✅ Type-safe
- ✅ Flexible
- ✅ Serializable

**Cons:**
- ⚠️ Requires component updates

### Strategy 3: Hybrid Approach (Recommended Overall)

Use markdown for simple content, config objects for complex elements.

---

## 📁 File Renaming Analysis

### Current Status: All .tsx Files Are Correct ✅

**Finding:** No .tsx files found without JSX

**Conclusion:** All current `.tsx` files contain JSX and should remain `.tsx`

### When to Rename .tsx → .ts

Only rename when **ALL** of these are true:
1. ✅ File contains NO JSX elements (`<...>`)
2. ✅ File contains NO JSX in any props
3. ✅ File is pure TypeScript/configuration

**Example of file that CAN be .ts:**
```typescript
// Pure config, no JSX
export const config = {
  title: "Hello",
  items: ["a", "b", "c"]
}
```

**Example of file that MUST stay .tsx:**
```typescript
// Has JSX in props
export const config = {
  title: "Hello",
  content: (<div>JSX here</div>) // ❌ JSX present
}
```

### Post-Migration Renaming

After completing SSG migration, check again:
```bash
find components -name "*.tsx" -type f -exec sh -c 'if ! grep -q "<" "$1"; then echo "$1"; fi' _ {} \;
```

---

## 🎯 Success Criteria

### Must Have
- [ ] Build succeeds without serialization errors
- [ ] All pages render correctly
- [ ] SSG generates static HTML for all pages
- [ ] No JSX in Props files
- [ ] Type-safe configuration objects

### Should Have
- [ ] Markdown support for FAQ answers
- [ ] Config objects for terminal output
- [ ] Config objects for decorations
- [ ] Documentation for new patterns

### Nice to Have
- [ ] CMS integration for markdown content
- [ ] Visual editor for config objects
- [ ] Automated migration tools

---

## 📊 Priority Matrix

### Immediate (This Week)
1. **MultiMachinePage** - Blocking build
2. **Build verification** - Ensure no other blockers

### Short Term (Next 2 Weeks)
1. **FAQ-heavy pages** - TermsPage, PrivacyPage, PricingPage
2. **Content-heavy pages** - FeaturesPage, ProvidersPage
3. **Developer pages** - DevelopersPage, CompliancePage

### Medium Term (Next Month)
1. **Medium priority pages** - 7 pages
2. **Low priority pages** - 4 pages
3. **Verification and testing**

---

## 🚀 Quick Start

### Step 1: Fix MultiMachinePage (Unblock Build)

```bash
# 1. Read the current state
cat frontend/apps/commercial/components/pages/MultiMachinePage/MultiMachinePageProps.tsx | grep -A 10 "subheadline:"

# 2. Create config types
# 3. Update Props file
# 4. Update Page file
# 5. Test build
cd frontend/apps/commercial && pnpm build
```

### Step 2: Choose Next Page

Based on:
- **Impact:** How many users see it?
- **Complexity:** How hard to migrate?
- **Dependencies:** Does it block other pages?

### Step 3: Follow Pattern

1. Identify JSX props
2. Create config types
3. Update Props file
4. Update Page file (if needed)
5. Test
6. Document

---

## 📚 Resources

### Documentation
- [ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md) - Aside migration example
- [HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md) - Component patterns
- [MIGRATION_EXECUTIVE_SUMMARY.md](./MIGRATION_EXECUTIVE_SUMMARY.md) - What we learned

### Tools
- TypeScript compiler for validation
- Next.js build for SSG testing
- Grep for finding JSX patterns

---

## ⚠️ Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation:** Test each page after migration

### Risk 2: Time Estimates Wrong
**Mitigation:** Start with critical page, reassess

### Risk 3: New Patterns Needed
**Mitigation:** Document as you go, create reusable components

### Risk 4: Content Loss
**Mitigation:** Backup Props files before migration

---

## 📈 Progress Tracking

Use this checklist:

```markdown
## Phase 1: Critical
- [ ] MultiMachinePage
- [ ] Build verification

## Phase 2: High Priority
- [ ] TermsPage (FAQ)
- [ ] PrivacyPage (FAQ)
- [ ] PricingPage (FAQ + visual)
- [ ] FeaturesPage (content)
- [ ] ProvidersPage (content)
- [ ] ResearchPage (decoration)
- [ ] DevelopersPage (terminal)
- [ ] CompliancePage (decoration)
- [ ] HomelabPage (terminal)

## Phase 3: Medium Priority
- [ ] 7 medium priority pages

## Phase 4: Low Priority
- [ ] 4 low priority pages

## Phase 5: Verification
- [ ] Full build test
- [ ] SSG output verification
- [ ] Performance testing
```

---

**Status:** 📋 PLAN READY  
**Next Action:** Fix MultiMachinePage to unblock build  
**Estimated Total Time:** 18 hours  
**Pages to Migrate:** 21 pages
