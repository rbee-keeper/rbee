# TEAM-423 Step 4: Fix FAQ Pages

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** PENDING  
**Priority:** 🟠 HIGH  
**Estimated Time:** 2 hours

---

## 🎯 Objective

Fix JSX props in FAQ-heavy pages by converting FAQ answers to markdown strings.

**Pages to fix:**
1. TermsPage (18 JSX props - FAQ answers)
2. PrivacyPage (18 JSX props - FAQ answers)
3. PricingPage (12 JSX props - FAQ + other)

**Total:** 48 JSX props

---

## 📋 Tasks

### Task 4.1: Fix TermsPage (40 min)

**Commands:**
```bash
cd frontend/apps/commercial/components/pages/TermsPage

# Find FAQ answers with JSX
grep -n "answer: (" TermsPageProps.tsx | wc -l

# View first few
grep -B 2 -A 10 "answer: (" TermsPageProps.tsx | head -50
```

**Pattern:**
```typescript
// ❌ WRONG
faqItems: [
  {
    question: "What are the terms?",
    answer: (
      <div className="prose">
        <p>These are the terms...</p>
        <ul>
          <li>Point 1</li>
          <li>Point 2</li>
        </ul>
      </div>
    )
  }
]

// ✅ CORRECT - Markdown string
faqItems: [
  {
    question: "What are the terms?",
    answer: "These are the terms...\n\n- Point 1\n- Point 2"
  }
]
```

**Action:**
1. Extract text content from each JSX answer
2. Convert to markdown format
3. Replace JSX with string
4. Test TypeScript compilation

### Task 4.2: Fix PrivacyPage (40 min)

**Same pattern as TermsPage:**

```bash
cd frontend/apps/commercial/components/pages/PrivacyPage

# Find FAQ answers
grep -n "answer: (" PrivacyPageProps.tsx | wc -l
```

**Action:** Apply same conversion as TermsPage.

### Task 4.3: Fix PricingPage (40 min)

**PricingPage has mixed JSX props:**

```bash
cd frontend/apps/commercial/components/pages/PricingPage

# Find all JSX props
grep -n ": (" PricingPageProps.tsx
```

**Fix:**
1. FAQ answers → markdown strings
2. `heading` prop → string or config
3. `visual` prop → config object
4. Any other JSX → appropriate format

**Pattern for visual:**
```typescript
// ❌ WRONG
visual: (
  <div className="chart">
    <ChartComponent />
  </div>
)

// ✅ CORRECT
visualConfig: {
  type: 'chart',
  component: 'ChartComponent'
}
```

---

## ✅ Success Criteria

### Per Page
- [ ] TermsPage: All 18 FAQ answers converted to markdown
- [ ] PrivacyPage: All 18 FAQ answers converted to markdown
- [ ] PricingPage: All 12 JSX props converted
- [ ] No `answer: (` patterns in any Props file
- [ ] TypeScript compiles for all pages

### Overall
- [ ] Build progresses past all FAQ pages
- [ ] No serialization errors
- [ ] FAQs render correctly (verify after build)

---

## 📝 Implementation Notes

**TEAM-423 signature:** Add to all modified files:
```typescript
// TEAM-423: Converted FAQ answers to markdown for SSG
```

**Files to modify:**
- `frontend/apps/commercial/components/pages/TermsPage/TermsPageProps.tsx`
- `frontend/apps/commercial/components/pages/PrivacyPage/PrivacyPageProps.tsx`
- `frontend/apps/commercial/components/pages/PricingPage/PricingPageProps.tsx`

**Markdown format:**
- Use `\n\n` for paragraphs
- Use `- ` for bullet points
- Use `**text**` for bold
- Keep it simple and readable

**FAQ Template Note:**
- Existing FAQ templates should already support markdown rendering
- If not, may need to update template (but prefer not to)

---

## 🔄 Handoff to Step 5

Once complete:
- Update TEAM_423_PROGRESS.md
- Mark Step 4 as complete
- Proceed to Step 5 (verification)

---

**Status:** READY TO IMPLEMENT  
**Prerequisite:** Steps 1-3 complete  
**Next:** Execute tasks 4.1-4.3
