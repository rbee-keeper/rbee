# STEP 01: Setup MDX Components Registry

**Estimated Time:** 1 hour  
**Priority:** 🔴 CRITICAL  
**Dependencies:** None

---

## Goal

Register existing `@rbee/ui` components for use in MDX files.

---

## What We're Reusing

**From `@rbee/ui/atoms`:**
- ✅ Alert, AlertTitle, AlertDescription
- ✅ Accordion, AccordionItem, AccordionTrigger, AccordionContent
- ✅ Tabs, TabsList, TabsTrigger, TabsContent
- ✅ Badge, Button
- ✅ Table, TableHeader, TableBody, TableRow, TableCell, TableHead
- ✅ Input
- ✅ Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
- ✅ CodeSnippet (already exists!)
- ✅ BrandMark, BrandWordmark
- ✅ Separator

**From `@rbee/ui/molecules`:**
- ✅ CodeBlock (already exists!)
- ✅ TerminalWindow (already exists!)
- ✅ ThemeToggle (already exists!)

---

## Task

Update `/app/docs/mdx-components.tsx` to register all components.

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/mdx-components.tsx`

```tsx
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'

// Atoms
import { 
  Alert, AlertTitle, AlertDescription,
  Accordion, AccordionItem, AccordionTrigger, AccordionContent,
  Tabs, TabsList, TabsTrigger, TabsContent,
  Badge, Button,
  Table, TableHeader, TableBody, TableRow, TableCell, TableHead,
  Input,
  Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter,
  CodeSnippet,
  BrandMark, BrandWordmark,
  Separator,
} from '@rbee/ui/atoms'

// Molecules
import { 
  CodeBlock,
  TerminalWindow,
  ThemeToggle,
} from '@rbee/ui/molecules'

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // Alerts
    Alert, AlertTitle, AlertDescription,
    
    // Accordion
    Accordion, AccordionItem, AccordionTrigger, AccordionContent,
    
    // Tabs
    Tabs, TabsList, TabsTrigger, TabsContent,
    
    // Code
    CodeSnippet,
    CodeBlock,
    TerminalWindow,
    
    // Tables
    Table, TableHeader, TableBody, TableRow, TableCell, TableHead,
    
    // Cards
    Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter,
    
    // UI Elements
    Badge, Button, Input, Separator,
    
    // Branding
    BrandMark, BrandWordmark,
    
    // Theme
    ThemeToggle,
    
    ...components,
  }
}
```

---

## Testing

```bash
cd frontend/apps/user-docs

# Type check
pnpm typecheck

# Should have no errors
```

**Test in MDX:**

Create test file: `/app/docs/test-components.mdx`

```mdx
# Component Test

<Alert>
  <AlertTitle>Test Alert</AlertTitle>
  <AlertDescription>This is a test</AlertDescription>
</Alert>

<CodeSnippet language="bash">
echo "Hello World"
</CodeSnippet>

<TerminalWindow title="Test">
$ echo "test"
test
</TerminalWindow>
```

Run dev server:
```bash
pnpm dev
# Visit http://localhost:7811/docs/test-components
```

---

## Acceptance Criteria

- [ ] No TypeScript errors
- [ ] All components import successfully
- [ ] Test MDX file renders all components
- [ ] No console errors

---

## Next Step

→ **STEP_02_CREATE_CALLOUT_WRAPPER.md**
