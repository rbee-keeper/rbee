# Fix Instructions for Next Team - SSG Serialization Issues

**Date:** 2025-11-08  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 2-3 hours  
**Difficulty:** Medium

---

## 🎯 Your Mission

Fix SSG serialization errors in comparison pages and high-priority pages to enable static site generation.

**Current Status:**
- ✅ MultiMachinePage fixed (3 JSX props converted)
- ❌ RbeeVsOllamaPage blocking build (component serialization issue)
- ⏳ 4 high-priority pages waiting (FeaturesPage, TermsPage, PrivacyPage, PricingPage)

---

## 📋 Step-by-Step Instructions

### PHASE 1: Fix RbeeVsOllamaPage (Critical Blocker)

**Time:** 1 hour  
**Goal:** Unblock the build

#### Step 1.1: Identify the Component Issue (15 min)

```bash
cd frontend/apps/commercial/components/pages/RbeeVsOllamaPage

# Check for component assignments (look for uppercase without <>)
grep -n ": [A-Z]" RbeeVsOllamaPageProps.tsx

# Check for Lucide icon imports
grep "from 'lucide-react'" RbeeVsOllamaPageProps.tsx

# Check for CodeBlock usage
grep -B 2 -A 5 "CodeBlock" RbeeVsOllamaPageProps.tsx
```

**What to look for:**
- Lines with `icon: Server` (should be `icon: 'Server'`)
- Lines with `component: SomeComponent` (should be config or JSX)
- Any uppercase identifiers assigned to props without `<>`

#### Step 1.2: Fix Icon References (15 min)

**Pattern to find:**
```typescript
// ❌ WRONG
icon: Server
icon: AlertTriangle
icon: X
```

**How to fix:**
```typescript
// ✅ CORRECT
icon: 'Server'
icon: 'AlertTriangle'
icon: 'X'
```

**Command to help:**
```bash
# Find all icon assignments
grep -n "icon: [A-Z]" RbeeVsOllamaPageProps.tsx
```

**Fix each one manually** - change component reference to string.

#### Step 1.3: Fix Any Component Props (15 min)

If you find any props that reference components:

```typescript
// ❌ WRONG
visual: CodeBlock
content: SomeComponent

// ✅ CORRECT - Option A: Render it
visual: <CodeBlock code="..." language="typescript" />

// ✅ BETTER - Option B: Config object
visualConfig: {
  type: 'code',
  code: '...',
  language: 'typescript'
}
```

#### Step 1.4: Test the Fix (15 min)

```bash
cd frontend/apps/commercial

# Check TypeScript
pnpm run typecheck | grep RbeeVsOllama

# Try build
pnpm build 2>&1 | tee /tmp/build.log

# Check if RbeeVsOllama error is gone
grep "rbee-vs-ollama" /tmp/build.log
```

**Success criteria:**
- No "rbee-vs-ollama" in build errors
- Build may fail on different page (that's progress!)
- TypeScript compiles without RbeeVsOllama errors

---

### PHASE 2: Fix Other Comparison Pages (30 min)

**Apply the same fixes to:**
- RbeeVsVllmPage
- RbeeVsTogetherAiPage
- RbeeVsRayKservePage

**For each page:**

```bash
cd components/pages/[PageName]

# 1. Check for issues
grep -n ": [A-Z]" [PageName]Props.tsx

# 2. Fix icon references (component → string)
# Edit file manually

# 3. Test
cd ../../..
pnpm run typecheck | grep [PageName]
```

**Batch command:**
```bash
cd frontend/apps/commercial/components/pages

for page in RbeeVsVllmPage RbeeVsTogetherAiPage RbeeVsRayKservePage; do
  echo "=== Checking $page ==="
  grep -n ": [A-Z]" $page/${page}Props.tsx | head -5
done
```

---

### PHASE 3: Fix High-Priority Pages (2 hours)

**Pages to fix:**
1. FeaturesPage (20 JSX props)
2. TermsPage (18 JSX props - FAQ answers)
3. PrivacyPage (18 JSX props - FAQ answers)
4. PricingPage (12 JSX props)

#### For FAQ-Heavy Pages (TermsPage, PrivacyPage, PricingPage)

**The Issue:**
```typescript
// ❌ WRONG - JSX in Props
faqItems: [
  {
    question: "...",
    answer: (
      <div className="prose">
        <p>Content here</p>
      </div>
    )
  }
]
```

**The Fix - Option A: Markdown (Recommended)**
```typescript
// ✅ CORRECT - Markdown string
faqItems: [
  {
    question: "...",
    answer: "Content here"  // Plain string or markdown
  }
]
```

**The Fix - Option B: HTML String**
```typescript
// ✅ ACCEPTABLE - HTML string
faqItems: [
  {
    question: "...",
    answerHTML: "<p>Content here</p>"
  }
]
```

**Steps:**
1. Open Props file
2. Find all `answer: (` patterns
3. Extract text content from JSX
4. Replace with plain string
5. Update FAQ template to render markdown (if needed)

#### For FeaturesPage (Content Props)

**The Issue:**
```typescript
// ❌ WRONG
content: (
  <div>
    <p>Feature description</p>
  </div>
)
```

**The Fix:**
```typescript
// ✅ CORRECT
content: "Feature description"

// OR for complex content
contentConfig: {
  type: 'prose',
  text: 'Feature description',
  highlights: ['key', 'points']
}
```

---

## 🔧 Tools & Commands

### Find All JSX in Props Files

```bash
cd frontend/apps/commercial/components/pages

# Find all ": (" patterns (likely JSX)
for dir in */; do
  page="${dir%/}"
  props_file=$(find "$dir" -name "*Props*.tsx" 2>/dev/null | head -1)
  if [ -f "$props_file" ]; then
    count=$(grep -c ": (" "$props_file" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
      echo "$page: $count JSX props"
    fi
  fi
done | sort -t: -k2 -rn
```

### Find Component References

```bash
# Find uppercase identifiers assigned to props
grep -n ": [A-Z][a-zA-Z]*[,}]" Props.tsx

# Find icon component references
grep -n "icon: [A-Z]" Props.tsx
```

### Test Build Incrementally

```bash
# Quick TypeScript check
pnpm run typecheck 2>&1 | grep -E "error TS|Found"

# Full build (may take time)
timeout 180 pnpm build 2>&1 | tee /tmp/build.log

# Check what failed
grep "Error occurred prerendering" /tmp/build.log
```

---

## 📊 Progress Tracking

Use this checklist:

```markdown
## Phase 1: Critical Blocker
- [ ] Identify component issue in RbeeVsOllamaPage
- [ ] Fix icon references
- [ ] Fix any component props
- [ ] Test build
- [ ] Verify error moved to different page

## Phase 2: Other Comparison Pages
- [ ] RbeeVsVllmPage
- [ ] RbeeVsTogetherAiPage
- [ ] RbeeVsRayKservePage
- [ ] Test build

## Phase 3: High Priority
- [ ] FeaturesPage (20 JSX props)
- [ ] TermsPage (18 FAQ answers)
- [ ] PrivacyPage (18 FAQ answers)
- [ ] PricingPage (12 JSX props)
- [ ] Final build test
```

---

## ✅ Success Criteria

### Per Page
- [ ] No component references in props (use strings)
- [ ] No JSX in props (use config objects or strings)
- [ ] TypeScript compiles
- [ ] Build progresses past this page

### Overall
- [ ] Build completes successfully
- [ ] All pages render correctly
- [ ] SSG generates static HTML
- [ ] No serialization errors

---

## ⚠️ Common Pitfalls

### Pitfall 1: Using 'force-dynamic'
```typescript
// ❌ DON'T DO THIS
export const dynamic = 'force-dynamic'
```
**Why:** Disables SSG. We want SSG to work, not bypass it.

### Pitfall 2: Keeping Component References
```typescript
// ❌ DON'T DO THIS
icon: Server  // This is a component, not a string
```
**Why:** Components can't be serialized for SSG.

### Pitfall 3: Complex JSX in Props
```typescript
// ❌ DON'T DO THIS
content: (
  <div className="complex">
    <Component />
  </div>
)
```
**Why:** JSX can't be serialized. Use config objects instead.

---

## 🎯 The Pattern to Follow

### From Our Successful Aside Migration

```typescript
// ✅ Props file - Config only
export const props = {
  asideConfig: {
    variant: 'icon',
    icon: 'Server',  // ← String, not component
    title: 'Title',
    subtitle: 'Subtitle'
  }
}

// ✅ Page file - Render from config
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...props} 
  aside={renderAside(props.asideConfig)}
/>
```

**Apply this pattern to:**
- FAQ answers → markdown strings
- Content props → config objects
- Icon props → strings
- Visual props → config objects

---

## 📚 Reference Documents

### Must Read
1. **[SSG_ROOT_CAUSE_ANALYSIS.md](./SSG_ROOT_CAUSE_ANALYSIS.md)** - Deep dive into the issue
2. **[ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md)** - Config pattern example
3. **[SSG_CRITICAL_HIGH_PRIORITY_TRACKER.md](./SSG_CRITICAL_HIGH_PRIORITY_TRACKER.md)** - Progress tracking

### For Reference
- [FULL_SSG_ENABLEMENT_PLAN.md](./FULL_SSG_ENABLEMENT_PLAN.md) - Complete plan
- [HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md) - Component patterns
- [MIGRATION_EXECUTIVE_SUMMARY.md](./MIGRATION_EXECUTIVE_SUMMARY.md) - Lessons learned

---

## 🚀 Getting Started

### Quick Start Commands

```bash
# 1. Navigate to workspace
cd /home/vince/Projects/llama-orch/frontend/apps/commercial

# 2. Start with critical blocker
cd components/pages/RbeeVsOllamaPage

# 3. Identify issues
grep -n ": [A-Z]" RbeeVsOllamaPageProps.tsx

# 4. Fix them (manually edit file)

# 5. Test
cd ../../..
pnpm run typecheck | grep RbeeVsOllama
pnpm build 2>&1 | head -50
```

---

## 💡 Tips for Success

### Tip 1: Work Incrementally
- Fix one page at a time
- Test after each fix
- Don't try to fix everything at once

### Tip 2: Use the Pattern
- Look at MultiMachinePage (already fixed)
- Look at aside migration (successful pattern)
- Apply the same approach

### Tip 3: Test Often
```bash
# Quick check
pnpm run typecheck | grep [PageName]

# Full check
pnpm build 2>&1 | tee /tmp/build.log
```

### Tip 4: Document as You Go
- Update SSG_CRITICAL_HIGH_PRIORITY_TRACKER.md
- Note any new patterns discovered
- Mark pages as complete

---

## 📞 If You Get Stuck

### Problem: Can't find the component reference
**Solution:** Use these greps:
```bash
grep -n ": [A-Z]" Props.tsx
grep -n "icon: [A-Z]" Props.tsx
grep -n "from 'lucide-react'" Props.tsx
```

### Problem: Build still fails after fix
**Solution:** Check the error message:
```bash
pnpm build 2>&1 | grep "Error occurred prerendering"
```
The error shows which page is blocking now.

### Problem: Don't know how to convert JSX to config
**Solution:** Look at the aside migration examples:
- See `TermsPageProps.tsx` (already has asideConfig)
- See `HeroAsides.tsx` (shows the pattern)
- Follow the same approach

---

## ✅ Final Checklist

Before marking as complete:

- [ ] All comparison pages fixed (4 pages)
- [ ] All high-priority pages fixed (4 pages)
- [ ] Build completes without errors
- [ ] All pages render correctly
- [ ] TypeScript compiles cleanly
- [ ] Progress tracker updated
- [ ] Patterns documented
- [ ] Changes committed

---

**Status:** 📋 INSTRUCTIONS READY  
**Next Team:** Start with Phase 1  
**Estimated Time:** 2-3 hours  
**Impact:** Enables SSG for entire app

**Good luck! You've got this! 🚀**
