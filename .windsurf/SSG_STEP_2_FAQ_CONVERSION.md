# SSG Step 2: Convert FAQ Answers to Markdown

**Step:** 2 of 5  
**Time:** 4-5 hours  
**Priority:** 🔴 CRITICAL  
**Status:** PENDING (requires Step 1 completion)

---

## 🎯 Goal

Convert ALL JSX FAQ answers to markdown strings to enable SSG.

---

## 📊 Scope

**Estimated:** 150+ JSX FAQ answers across 10+ pages

**Pages with FAQ JSX:**
- PrivacyPage (18 answers)
- TermsPage (17 answers, hero already fixed)
- PricingPage (12 answers)
- FeaturesPage (some FAQ items)
- Other pages (various)

---

## 📋 Conversion Strategy

### Option A: Plain Text (Fastest)
**Time:** 3-4 hours  
**Pros:** Quick, simple  
**Cons:** Loses formatting

```typescript
// Before
answer: (
  <div className="prose">
    <p>This is the answer.</p>
    <ul>
      <li>Point 1</li>
      <li>Point 2</li>
    </ul>
  </div>
)

// After
answer: "This is the answer.\n\n- Point 1\n- Point 2"
```

### Option B: Markdown (Recommended)
**Time:** 4-5 hours  
**Pros:** Preserves formatting, clean  
**Cons:** Requires markdown renderer

```typescript
// Before
answer: (
  <div className="prose">
    <p><strong>Important:</strong> This is the answer.</p>
    <ul>
      <li>Point 1</li>
      <li>Point 2</li>
    </ul>
  </div>
)

// After
answer: "**Important:** This is the answer.\n\n- Point 1\n- Point 2"
```

### Option C: HTML Strings (Fallback)
**Time:** 2-3 hours  
**Pros:** Fastest, preserves exact formatting  
**Cons:** Less clean, harder to maintain

```typescript
// Before
answer: (<div className="prose"><p>Text</p></div>)

// After
answer: '<div class="prose"><p>Text</p></div>'
```

---

## 📋 Tasks

### Task 2.1: Choose Conversion Strategy (15 min)

**Decision:** Pick Option A, B, or C based on:
- Time available
- Formatting requirements
- Maintenance preferences

**Recommendation:** Option B (Markdown)

### Task 2.2: Update FAQ Template (30 min)

If using markdown, update FAQ template to render it:

```typescript
// In FAQTemplate.tsx
import ReactMarkdown from 'react-markdown'

// In render
{typeof item.answer === 'string' ? (
  <ReactMarkdown className="prose">{item.answer}</ReactMarkdown>
) : (
  item.answer
)}
```

**Files to update:**
- `components/templates/FAQTemplate/FAQTemplate.tsx`

### Task 2.3: Convert PrivacyPage (1 hour)

**File:** `components/pages/PrivacyPage/PrivacyPageProps.tsx`  
**FAQ Answers:** 18

**Process:**
1. Open file
2. Find each `answer: (` pattern
3. Extract text content from JSX
4. Convert to markdown string
5. Replace JSX with string
6. Test compilation

**Example:**
```typescript
// Before (lines 70-86)
answer: (
  <div className="space-y-4">
    <p>
      This Privacy Policy applies to all users of rbee software and services.
    </p>
    <p><strong>Scope:</strong></p>
    <ul className="list-disc space-y-2 pl-6">
      <li>rbee open-source software (GPL-3.0-or-later)</li>
      <li>rbee.ai website and documentation</li>
    </ul>
  </div>
)

// After
answer: "This Privacy Policy applies to all users of rbee software and services.\n\n**Scope:**\n\n- rbee open-source software (GPL-3.0-or-later)\n- rbee.ai website and documentation"
```

### Task 2.4: Convert TermsPage (1 hour)

**File:** `components/pages/TermsPage/TermsPageProps.tsx`  
**FAQ Answers:** 17 (hero already fixed)

Same process as PrivacyPage.

### Task 2.5: Convert PricingPage (45 min)

**File:** `components/pages/PricingPage/PricingPageProps.tsx`  
**FAQ Answers:** 12

Same process as PrivacyPage.

### Task 2.6: Convert Remaining Pages (1-2 hours)

Convert FAQ answers in:
- FeaturesPage
- Any other pages with FAQ JSX

### Task 2.7: Test All Conversions (30 min)

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial

# TypeScript check
pnpm run typecheck

# Build test (with force-dynamic still enabled)
pnpm build
```

---

## 🔧 Conversion Tools

### Manual Conversion Template

For each FAQ answer:

1. **Extract text:**
   - Copy JSX content
   - Remove all HTML tags
   - Preserve structure with markdown

2. **Convert formatting:**
   - `<strong>` → `**text**`
   - `<ul><li>` → `- item`
   - `<p>` → paragraph breaks (`\n\n`)
   - `<a href="">` → `[text](url)`
   - `<code>` → `` `code` ``

3. **Escape quotes:**
   - Replace `"` with `\"`
   - Or use single quotes for string

4. **Test:**
   - Verify markdown renders correctly
   - Check no syntax errors

### Batch Conversion Script (Optional)

If many similar FAQ items, create a script:

```python
#!/usr/bin/env python3
import re

def jsx_to_markdown(jsx):
    # Remove className attributes
    text = re.sub(r'\s*className="[^"]*"', '', jsx)
    # Convert tags
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text)
    text = re.sub(r'<li>(.*?)</li>', r'- \1', text)
    text = re.sub(r'</?(?:ul|div|p)(?:\s[^>]*)?>', '', text)
    # Clean whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()
```

---

## ✅ Completion Criteria

- [ ] All FAQ answers converted to strings
- [ ] No `answer: (` patterns in any props file
- [ ] FAQ template updated (if using markdown)
- [ ] TypeScript compiles
- [ ] Build succeeds (with force-dynamic)
- [ ] FAQs render correctly in browser

---

## 📊 Progress Tracking

Create `FAQ_CONVERSION_PROGRESS.md`:

```markdown
## FAQ Conversion Progress

- [ ] PrivacyPage (0/18)
- [ ] TermsPage (0/17)
- [ ] PricingPage (0/12)
- [ ] FeaturesPage (0/?)
- [ ] Other pages (0/?)

**Total:** 0/150+ converted
```

---

## 🔄 Handoff to Step 3

Once all FAQ answers are converted:
1. Verify no `answer: (` patterns remain
2. Test build
3. Proceed to Step 3 (Component References)

---

**Status:** PENDING  
**Prerequisite:** Step 1 complete  
**Next:** Convert FAQ answers to markdown
