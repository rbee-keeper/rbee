# SSG Step 1: Audit All Props Files

**Step:** 1 of 5  
**Time:** 1 hour  
**Priority:** 🔴 CRITICAL  
**Status:** READY TO EXECUTE

---

## 🎯 Goal

Identify ALL JSX in props files across the entire commercial frontend.

---

## 📋 Tasks

### Task 1.1: Scan for JSX Patterns (15 min)

**Command:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages

# Find all JSX in props files
for dir in */; do
  page="${dir%/}"
  props_file="$dir/${page}Props.tsx"
  if [ -f "$props_file" ]; then
    jsx_count=$(grep -c ": (" "$props_file" 2>/dev/null || echo "0")
    comp_imports=$(grep "^import { [A-Z]" "$props_file" 2>/dev/null | grep -v "^import type" | wc -l)
    if [ "$jsx_count" -gt 0 ] || [ "$comp_imports" -gt 0 ]; then
      echo "$page: JSX=$jsx_count, ComponentImports=$comp_imports"
    fi
  fi
done | sort -t: -k2 -rn
```

### Task 1.2: Create Detailed Inventory (30 min)

For each page with JSX, document:
1. Page name
2. Number of JSX props
3. Type of JSX (FAQ, hero, visual, etc.)
4. Component imports
5. Estimated fix time

**Output:** Create `SSG_AUDIT_RESULTS.md`

### Task 1.3: Categorize by Fix Type (15 min)

Group pages by fix strategy:
- **FAQ Conversion** - Pages with FAQ JSX answers
- **Component Refs** - Pages with component references
- **Hero/Visual** - Pages with JSX in hero/visual props
- **Mixed** - Pages with multiple types

---

## 📊 Expected Results

### Pages with JSX (from previous scan)

| Page | JSX Props | Component Imports | Type |
|------|-----------|-------------------|------|
| FeaturesPage | 20 | 4 | Mixed |
| PrivacyPage | 18 | 0 | FAQ |
| TermsPage | 17 | 0 | FAQ (partial fix) |
| PricingPage | 12 | 0 | FAQ |
| DevelopersPage | 9 | 4 | Mixed |
| ResearchPage | 9 | 3 | Mixed |
| CompliancePage | 8 | 2 | Mixed |
| HomelabPage | 6 | 0 | Mixed |
| CommunityPage | 5 | 0 | Mixed |
| EnterprisePage | 5 | 4 | Mixed |
| +10 more | 30+ | 10+ | Various |

**Total:** ~150+ JSX props across 20+ pages

---

## 📝 Audit Template

For each page, document:

```markdown
### [PageName]

**File:** `components/pages/[PageName]/[PageName]Props.tsx`  
**JSX Props:** [count]  
**Component Imports:** [count]

**JSX Breakdown:**
- FAQ answers: [count]
- Hero subcopy: [yes/no]
- Visual props: [count]
- Component refs: [count]
- Other: [count]

**Fix Strategy:** [FAQ/Component/Hero/Mixed]  
**Estimated Time:** [hours]  
**Priority:** [Critical/High/Medium/Low]

**Example JSX:**
```typescript
// Show 1-2 examples of the JSX found
```

**Proposed Fix:**
```typescript
// Show how it should be fixed
```
```

---

## 🔍 Detailed Scan Commands

### Find All FAQ JSX
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages
grep -r "answer: (" . --include="*Props.tsx" | wc -l
```

### Find All Component Imports
```bash
grep -r "^import { [A-Z]" . --include="*Props.tsx" | grep -v "^import type" | wc -l
```

### Find All Hero JSX
```bash
grep -r "subcopy: (" . --include="*Props.tsx" | wc -l
```

### Find All Visual JSX
```bash
grep -r "visual: (" . --include="*Props.tsx" | wc -l
grep -r "decoration: (" . --include="*Props.tsx" | wc -l
```

---

## ✅ Completion Criteria

- [ ] All props files scanned
- [ ] JSX count documented for each page
- [ ] Pages categorized by fix type
- [ ] Audit results saved to `SSG_AUDIT_RESULTS.md`
- [ ] Fix time estimates calculated
- [ ] Priority order established

---

## 📊 Output Document

Create `SSG_AUDIT_RESULTS.md` with:

1. **Executive Summary**
   - Total pages with JSX
   - Total JSX props to fix
   - Total estimated time

2. **Detailed Inventory**
   - Per-page breakdown
   - Fix strategy for each
   - Priority ranking

3. **Fix Categories**
   - FAQ pages (list)
   - Component ref pages (list)
   - Hero/Visual pages (list)
   - Mixed pages (list)

4. **Recommended Fix Order**
   - Phase 1 pages (critical)
   - Phase 2 pages (high)
   - Phase 3 pages (medium)
   - Phase 4 pages (low)

---

## 🔄 Handoff to Step 2

Once audit is complete:
1. Review `SSG_AUDIT_RESULTS.md`
2. Confirm fix strategy for each page
3. Proceed to Step 2 (FAQ Conversion)

---

**Status:** READY TO EXECUTE  
**Next:** Run scan commands and create audit document  
**Output:** `SSG_AUDIT_RESULTS.md`
