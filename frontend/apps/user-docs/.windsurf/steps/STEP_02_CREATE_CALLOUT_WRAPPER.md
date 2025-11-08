# STEP 02: Create Callout Wrapper Component

**Estimated Time:** 1 hour  
**Priority:** 🔴 CRITICAL  
**Dependencies:** STEP_01

---

## Goal

Create MDX-friendly `<Callout>` wrapper around existing `Alert` component.

---

## What We're Reusing

- ✅ `Alert`, `AlertTitle`, `AlertDescription` from `@rbee/ui/atoms`
- ✅ Icons from `lucide-react`

---

## Task

Create a simple wrapper that makes Alert easier to use in MDX.

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/_components/Callout.tsx`

```tsx
import { Alert, AlertTitle, AlertDescription } from '@rbee/ui/atoms'
import { Info, AlertTriangle, XCircle, CheckCircle } from 'lucide-react'

interface CalloutProps {
  variant?: 'info' | 'warning' | 'error' | 'success'
  title?: string
  children: React.ReactNode
}

const icons = {
  info: Info,
  warning: AlertTriangle,
  error: XCircle,
  success: CheckCircle,
}

const variantMap = {
  info: 'default',
  warning: 'default', // Alert doesn't have warning, use default
  error: 'destructive',
  success: 'default', // Alert doesn't have success, use default
} as const

export function Callout({ 
  variant = 'info', 
  title, 
  children 
}: CalloutProps) {
  const Icon = icons[variant]
  
  return (
    <Alert variant={variantMap[variant]} className="my-6">
      <Icon className="h-4 w-4" />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}
```

---

## Register in MDX

**File:** `/frontend/apps/user-docs/app/docs/mdx-components.tsx`

Add import:
```tsx
import { Callout } from './_components/Callout'
```

Add to exports:
```tsx
export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // ... existing exports ...
    
    // Custom wrappers
    Callout,
    
    ...components,
  }
}
```

---

## Testing

**Test file:** `/app/docs/test-components.mdx`

```mdx
# Callout Test

<Callout variant="info" title="Information">
This is an info callout.
</Callout>

<Callout variant="warning" title="Warning">
This is a warning callout.
</Callout>

<Callout variant="error" title="Error">
This is an error callout.
</Callout>

<Callout variant="success" title="Success">
This is a success callout.
</Callout>

<Callout variant="info">
Callout without title.
</Callout>
```

Run:
```bash
pnpm dev
# Visit http://localhost:7811/docs/test-components
```

---

## Acceptance Criteria

- [ ] All 4 variants render correctly
- [ ] Icons display
- [ ] Title is optional
- [ ] Spacing looks good (my-6)
- [ ] Dark mode works

---

## Next Step

→ **STEP_03_CREATE_CODETABS_WRAPPER.md**
