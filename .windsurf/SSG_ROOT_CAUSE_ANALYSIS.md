# SSG Root Cause Analysis - RbeeVsOllamaPage Blocker

**Date:** 2025-11-08 01:45 AM  
**Status:** 🔍 DEEP ANALYSIS  
**Priority:** 🔴 CRITICAL - BLOCKING BUILD

---

## 🎯 The Problem

**Error:**
```
Error occurred prerendering page "/compare/rbee-vs-ollama"
Error: Functions cannot be passed directly to Client Components unless you explicitly expose it by marking it with "use server".
{$$typeof: ..., render: function, displayName: ...}
```

**What This Means:**
Next.js is trying to statically generate (SSG) the page, but somewhere in the component tree, a **React component** (which contains functions) is being passed as a prop instead of being rendered.

---

## 🔍 Investigation Results

### ✅ What's NOT the Problem

1. **Props File** - No JSX in Props file (checked)
2. **Page Component** - Has `'use client'` directive
3. **SEO Functions** - Return plain objects, not components
4. **Route Config** - Standard Next.js app route

### ❌ What IS the Problem

The error `{$$typeof: ..., render: function, displayName: ...}` indicates a **React component object** is being passed somewhere it shouldn't be.

**Most Likely Causes:**

1. **Template Component Issue**
   - One of the templates (HeroTemplate, ComparisonTemplate, etc.) is receiving a component as a prop
   - Should receive JSX element or config object instead

2. **CodeBlock Import**
   - Line 4 of Props file: `import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'`
   - If CodeBlock is being passed as a prop (not rendered), this causes the error

3. **Icon Components**
   - If Lucide icons are imported as components and passed as props
   - Should be passed as strings or rendered inline

---

## 🔬 Detailed Analysis

### File Structure

```
app/compare/rbee-vs-ollama/
├── page.tsx                    ← App route (no 'use client')
└── (imports RbeeVsOllamaPage)

components/pages/RbeeVsOllamaPage/
├── RbeeVsOllamaPage.tsx       ← Has 'use client'
└── RbeeVsOllamaPageProps.tsx  ← Props definitions
```

### The Issue

Even though `RbeeVsOllamaPage.tsx` has `'use client'`, the **app route** (`page.tsx`) does NOT.

**Next.js Behavior:**
1. App route tries to SSG (static generation)
2. Imports `RbeeVsOllamaPage` component
3. Tries to serialize props for SSG
4. Encounters a React component in props
5. **FAILS** with serialization error

### Why 'use client' Doesn't Help

The `'use client'` directive only affects the component itself, not the app route that imports it. The app route still tries to SSG, which means it still tries to serialize all props.

---

## 🎯 Root Cause

**The actual problem:** Something in the component tree is passing a **React component** as a prop instead of:
- Rendering it (`<Component />`)
- Passing a config object
- Passing a string

**Where to look:**

1. **Check all template props** for component references
2. **Check icon props** - should be strings, not components
3. **Check CodeBlock usage** - should be rendered, not passed
4. **Check any `content` or `visual` props** - should be config, not components

---

## 🔧 How to Find the Exact Issue

### Step 1: Check Template Props

```bash
# Look for component assignments in Props file
grep -n "CodeBlock\|Icon\|Component" RbeeVsOllamaPageProps.tsx
```

### Step 2: Check Icon Usage

```bash
# Look for icon imports
grep -n "from 'lucide-react'" RbeeVsOllamaPageProps.tsx
```

### Step 3: Check Template Definitions

```bash
# Check what templates expect
grep -A 5 "export.*Template" components/templates/*/index.tsx
```

---

## 🛠️ The Correct Fix

### Option 1: Fix Component Passing (CORRECT)

**If the issue is a component being passed as a prop:**

```typescript
// ❌ WRONG - Passing component
content: CodeBlock

// ✅ RIGHT - Rendering component
content: <CodeBlock code="..." language="typescript" />

// ✅ BETTER - Config object
contentConfig: {
  type: 'code',
  code: '...',
  language: 'typescript'
}
```

### Option 2: Fix Icon Passing (CORRECT)

```typescript
// ❌ WRONG - Passing component
icon: Server

// ✅ RIGHT - Passing string
icon: 'Server'
```

### Option 3: Add Dynamic Export (TEMPORARY - NOT RECOMMENDED)

```typescript
// app/compare/rbee-vs-ollama/page.tsx
export const dynamic = 'force-dynamic'
```

**Why NOT recommended:** This disables SSG entirely. We want SSG to work.

---

## 📋 Fix Instructions for Next Team

### Phase 1: Identify the Exact Issue (15 min)

1. **Search for component assignments:**
   ```bash
   cd frontend/apps/commercial/components/pages/RbeeVsOllamaPage
   grep -n ": [A-Z]" RbeeVsOllamaPageProps.tsx
   ```

2. **Check for Lucide imports:**
   ```bash
   grep "from 'lucide-react'" RbeeVsOllamaPageProps.tsx
   ```

3. **Check for CodeBlock usage:**
   ```bash
   grep -A 3 "CodeBlock" RbeeVsOllamaPageProps.tsx
   ```

4. **Look for the pattern:**
   - Component name without `<>`
   - Assigned to a prop
   - Not rendered

### Phase 2: Fix the Issue (30 min)

**For each component found:**

1. **If it's an icon:**
   ```typescript
   // Before
   icon: Server
   
   // After
   icon: 'Server'
   ```

2. **If it's CodeBlock or similar:**
   ```typescript
   // Before
   visual: CodeBlock
   
   // After - Option A: Render it
   visual: <CodeBlock code="..." />
   
   // After - Option B: Config object (BETTER)
   visualConfig: {
     type: 'code',
     code: '...',
     language: 'typescript'
   }
   ```

3. **If it's a custom component:**
   - Create a config object
   - Update template to render from config
   - Follow the aside pattern we established

### Phase 3: Apply to All Comparison Pages (1 hour)

**Pages to check:**
- RbeeVsOllamaPage ← Current blocker
- RbeeVsVllmPage
- RbeeVsTogetherAiPage
- RbeeVsRayKservePage

**For each page:**
1. Run Phase 1 checks
2. Apply Phase 2 fixes
3. Test build
4. Verify page renders

### Phase 4: Verify (15 min)

```bash
# Test TypeScript
cd frontend/apps/commercial
pnpm run typecheck

# Test build
pnpm build

# Should complete without errors
```

---

## 🎯 Success Criteria

- [ ] Build completes without errors
- [ ] All comparison pages render correctly
- [ ] No component objects in props
- [ ] All icons are strings
- [ ] All complex content uses config objects
- [ ] SSG works (pages in .next/server/app/)

---

## 📊 Pattern to Follow

### The Correct Pattern (From Aside Migration)

```typescript
// Props file (.tsx)
export const props = {
  // ✅ Config object (serializable)
  asideConfig: {
    variant: 'icon',
    icon: 'Server',  // ← String, not component
    title: 'Title'
  }
}

// Page file (.tsx)
import { renderAside } from '../../organisms/HeroAsides'

<HeroTemplate 
  {...props} 
  aside={renderAside(props.asideConfig)}  // ← Render from config
/>
```

### Apply This Pattern

For any complex content:
1. Create config type
2. Store config in Props file
3. Create renderer function
4. Render in Page file

---

## ⚠️ Common Mistakes to Avoid

### Mistake 1: Quick Fix with 'force-dynamic'
```typescript
// ❌ DON'T DO THIS
export const dynamic = 'force-dynamic'
```
**Why:** Disables SSG. We want SSG to work.

### Mistake 2: Keeping Component References
```typescript
// ❌ DON'T DO THIS
icon: Server  // Component reference
```
**Why:** Can't be serialized for SSG.

### Mistake 3: Inline JSX in Props
```typescript
// ❌ DON'T DO THIS
content: (<div>...</div>)
```
**Why:** JSX can't be serialized for SSG.

---

## 📚 Related Documentation

- [ASIDES_SOLUTION_INDEX.md](./ASIDES_SOLUTION_INDEX.md) - Config pattern example
- [HERO_ASIDES_GUIDE.md](./HERO_ASIDES_GUIDE.md) - Renderer pattern
- [MIGRATION_EXECUTIVE_SUMMARY.md](./MIGRATION_EXECUTIVE_SUMMARY.md) - What we learned

---

## 🚀 Next Steps

1. **Immediate:** Run Phase 1 to identify exact issue
2. **Next:** Apply Phase 2 fix
3. **Then:** Apply to all comparison pages (Phase 3)
4. **Finally:** Verify build works (Phase 4)

---

**Status:** 📋 ANALYSIS COMPLETE  
**Next Action:** Phase 1 - Identify exact component being passed  
**Estimated Time:** 1 hour total  
**Impact:** Unblocks build for all comparison pages
