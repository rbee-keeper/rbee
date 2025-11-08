# SSG Step 3: Fix Component References

**Step:** 3 of 5  
**Time:** 2 hours  
**Priority:** 🟠 HIGH  
**Status:** PENDING (requires Step 2 completion)

---

## 🎯 Goal

Convert all component references in props to strings or config objects.

---

## 📊 Scope

**Component Issues:**
1. Icon component references (e.g., `icon: Server`)
2. Component imports in props files
3. Component props (e.g., `visual: CodeBlock`)

**Already Fixed:**
- ✅ RbeeVsOllamaPage - CodeBlock import removed
- ✅ ProvidersPage - 6 icon references fixed
- ✅ PricingPage - PricingScaleVisual import removed

---

## 📋 Tasks

### Task 3.1: Find All Component References (30 min)

**Scan for component references:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages

# Find icon component references
grep -r "icon: [A-Z]" . --include="*Props.tsx" | grep -v "icon: '[A-Z]"

# Find visual component references
grep -r "visual: [A-Z]" . --include="*Props.tsx"

# Find component imports
grep -r "^import { [A-Z]" . --include="*Props.tsx" | grep -v "^import type"
```

### Task 3.2: Fix Icon References (45 min)

**Pattern:**
```typescript
// ❌ WRONG
icon: Server
icon: AlertTriangle
kickerIcon: <Zap className="h-4 w-4" />

// ✅ CORRECT
icon: 'Server'
icon: 'AlertTriangle'
kickerIcon: 'Zap'
```

**Files to check:**
- CommunityPage
- EducationPage
- EnterprisePage
- DevelopersPage
- ResearchPage
- CompliancePage
- SecurityPage
- All feature pages

**Process:**
1. Find all icon references
2. Convert component to string
3. Remove unused imports
4. Test compilation

### Task 3.3: Fix Component Imports (30 min)

**Check these imports:**
- `CodeBlock` - Used or unused?
- `GPUUtilizationBar` - Used or unused?
- `TerminalWindow` - Used or unused?
- `Badge` - Used or unused?
- `CacheLayer`, `DiagnosticGrid`, etc. - Used or unused?

**For each import:**
1. Search if it's actually used in the file
2. If used in JSX, keep it (will fix in Step 4)
3. If unused, remove the import

**Command to check usage:**
```bash
# Example: Check if CodeBlock is used
grep "CodeBlock" FeaturesPageProps.tsx | grep -v "^import"
```

### Task 3.4: Fix Visual/Component Props (45 min)

**Pattern:**
```typescript
// ❌ WRONG
visual: CodeBlock
decoration: <SomeComponent />

// ✅ CORRECT - Option A: Config object
visualConfig: {
  type: 'code',
  code: '...',
  language: 'typescript'
}

// ✅ CORRECT - Option B: Render in page component
// Keep prop as config, render JSX in page.tsx
```

**Files with component props:**
- FeaturesPage (CodeBlock, GPUUtilizationBar, TerminalWindow)
- Any pages with visual/decoration props

---

## 🔧 Fix Strategies

### Strategy A: String Conversion (Icons)

**For icon props:**
```typescript
// Before
import { Server, Zap } from 'lucide-react'

export const props = {
  icon: Server,
  kickerIcon: <Zap className="h-4 w-4" />
}

// After
export const props = {
  icon: 'Server',
  kickerIcon: 'Zap'
}
```

**Template handles string → component:**
```typescript
// In template
import * as Icons from 'lucide-react'

const IconComponent = Icons[iconName]
return <IconComponent />
```

### Strategy B: Config Objects (Complex Components)

**For complex components:**
```typescript
// Before
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'

export const props = {
  content: (
    <CodeBlock
      code="const x = 1"
      language="typescript"
    />
  )
}

// After
export const props = {
  contentConfig: {
    type: 'code',
    code: 'const x = 1',
    language: 'typescript'
  }
}

// In page.tsx
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'

<Template
  {...props}
  content={<CodeBlock {...props.contentConfig} />}
/>
```

### Strategy C: Remove Unused Imports

**For unused components:**
```typescript
// Before
import { Badge } from '@rbee/ui/atoms'
// ... Badge never used in file

// After
// Just remove the import
```

---

## 📝 Detailed Fix List

### FeaturesPage
**File:** `components/pages/FeaturesPage/FeaturesPageProps.tsx`

**Issues:**
- `CodeBlock` - Used in content props
- `GPUUtilizationBar` - Used in content props
- `TerminalWindow` - Used in content props
- `Badge`, `CacheLayer`, `DiagnosticGrid`, etc. - Check usage

**Fix:**
1. Check if each is used in JSX content
2. If yes, convert to config object
3. If no, remove import

### EducationPage
**File:** `components/pages/EducationPage/EducationPageProps.tsx`

**Issues:**
- Component imports (5 total)
- Icon references

**Fix:**
1. Convert icon references to strings
2. Remove unused imports

### EnterprisePage
**File:** `components/pages/EnterprisePage/EnterprisePageProps.tsx`

**Issues:**
- Component imports (4 total)
- Icon references

**Fix:**
1. Convert icon references to strings
2. Remove unused imports

---

## ✅ Completion Criteria

- [ ] All icon references converted to strings
- [ ] All unused component imports removed
- [ ] All component props converted to config objects
- [ ] No component references in props files
- [ ] TypeScript compiles
- [ ] Build succeeds (with force-dynamic)

---

## 🔄 Handoff to Step 4

Once all component references are fixed:
1. Verify no component references remain
2. Test build
3. Proceed to Step 4 (Hero/Visual Props)

---

**Status:** PENDING  
**Prerequisite:** Step 2 complete  
**Next:** Fix all component references
