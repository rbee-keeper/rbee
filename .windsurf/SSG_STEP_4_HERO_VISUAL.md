# SSG Step 4: Fix Hero and Visual Props

**Step:** 4 of 5  
**Time:** 1-2 hours  
**Priority:** 🟠 HIGH  
**Status:** PENDING (requires Step 3 completion)

---

## 🎯 Goal

Convert all remaining JSX in hero sections and visual props to serializable format.

---

## 📊 Scope

**Remaining JSX:**
1. Hero subcopy with JSX
2. Visual/decoration props with JSX
3. Content props with JSX (not FAQ)
4. Any other JSX not covered in Steps 2-3

**Already Fixed:**
- ✅ TermsPage hero subcopy

---

## 📋 Tasks

### Task 4.1: Find Remaining JSX (15 min)

**Scan for all JSX patterns:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages

# Find subcopy JSX
grep -r "subcopy: (" . --include="*Props.tsx"

# Find visual JSX
grep -r "visual: (" . --include="*Props.tsx"

# Find decoration JSX
grep -r "decoration: (" . --include="*Props.tsx"

# Find content JSX (non-FAQ)
grep -r "content: (" . --include="*Props.tsx"

# Find any remaining ": (" patterns
grep -r ": (" . --include="*Props.tsx" | wc -l
```

### Task 4.2: Fix Hero Subcopy (30 min)

**Pattern:**
```typescript
// ❌ WRONG
subcopy: (
  <div className="space-y-2">
    <p>Main text here.</p>
    <p className="text-sm">
      <strong>Last Updated:</strong> Date
    </p>
  </div>
)

// ✅ CORRECT
subcopy: "Main text here.\n\nLast Updated: Date"
```

**Files to check:**
- All pages with hero sections
- Legal pages (privacy, terms)
- Feature pages

**Process:**
1. Find all `subcopy: (` patterns
2. Extract text content
3. Convert to plain string
4. Test rendering

### Task 4.3: Fix Visual/Decoration Props (45 min)

**Pattern:**
```typescript
// ❌ WRONG
decoration: (
  <div className="absolute inset-0 opacity-15">
    <DistributedNodes />
  </div>
)

// ✅ CORRECT - Option A: Config object
decorationConfig: {
  type: 'distributed-nodes',
  className: 'absolute inset-0 opacity-15'
}

// ✅ CORRECT - Option B: Render in page component
// Move JSX to page.tsx, keep config in props
```

**Files with visual/decoration:**
- FeaturesPage (multiple decorations)
- Any pages with background decorations

**Decision:** Choose Option A or B based on complexity

### Task 4.4: Fix Content Props (30 min)

**For non-FAQ content props:**

```typescript
// ❌ WRONG
content: (
  <div className="prose">
    <p>Complex content with formatting</p>
  </div>
)

// ✅ CORRECT - Option A: Markdown
content: "Complex content with formatting"

// ✅ CORRECT - Option B: Config
contentConfig: {
  type: 'prose',
  text: 'Complex content with formatting'
}
```

**Files to check:**
- FeaturesPage
- Any pages with content props

---

## 🔧 Fix Strategies

### Strategy A: Plain Strings (Simple Content)

**Use for:**
- Simple text
- No complex formatting
- No nested elements

```typescript
// Before
subcopy: (<p>Simple text</p>)

// After
subcopy: "Simple text"
```

### Strategy B: Markdown Strings (Formatted Content)

**Use for:**
- Bold/italic text
- Lists
- Links
- Code snippets

```typescript
// Before
subcopy: (
  <div>
    <p><strong>Important:</strong> Text here</p>
    <ul>
      <li>Point 1</li>
    </ul>
  </div>
)

// After
subcopy: "**Important:** Text here\n\n- Point 1"
```

### Strategy C: Config Objects (Complex Components)

**Use for:**
- Custom components
- Complex layouts
- Dynamic content

```typescript
// Before
decoration: (<CustomComponent prop="value" />)

// After
decorationConfig: {
  type: 'custom',
  prop: 'value'
}

// Render in template or page
```

### Strategy D: Move to Page Component

**Use for:**
- Very complex JSX
- One-off cases
- Hard to serialize

```typescript
// In props file
export const props = {
  // No JSX, just config
  showDecoration: true
}

// In page.tsx
<Template
  {...props}
  decoration={props.showDecoration && <ComplexJSX />}
/>
```

---

## 📝 Detailed Fix List

### Pages with Hero Subcopy JSX
- [ ] Check all hero sections
- [ ] Convert to plain strings
- [ ] Test rendering

### Pages with Visual/Decoration JSX
- [ ] FeaturesPage decorations
- [ ] Any background elements
- [ ] Convert to config or move to page

### Pages with Content JSX
- [ ] FeaturesPage content props
- [ ] Any other content props
- [ ] Convert to markdown or config

---

## ✅ Completion Criteria

- [ ] No `subcopy: (` patterns in props files
- [ ] No `visual: (` patterns in props files
- [ ] No `decoration: (` patterns in props files
- [ ] No `content: (` patterns in props files (except FAQ, already fixed)
- [ ] All JSX converted to serializable format
- [ ] TypeScript compiles
- [ ] Build succeeds (with force-dynamic)
- [ ] All pages render correctly

---

## 🔍 Final Verification

**Run comprehensive scan:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages

# Count all remaining ": (" patterns
total=$(grep -r ": (" . --include="*Props.tsx" | wc -l)
echo "Remaining JSX patterns: $total"

# Should be 0 after this step!
```

---

## 🔄 Handoff to Step 5

Once all JSX is converted:
1. Verify 0 JSX patterns remain
2. Test build with force-dynamic
3. Proceed to Step 5 (Remove force-dynamic and restore SSG)

---

**Status:** PENDING  
**Prerequisite:** Step 3 complete  
**Next:** Fix all remaining JSX in hero/visual props
