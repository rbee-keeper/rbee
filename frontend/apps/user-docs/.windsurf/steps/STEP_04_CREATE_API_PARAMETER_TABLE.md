# STEP 04: Create APIParameterTable Component

**Estimated Time:** 2 hours  
**Priority:** 🔴 CRITICAL  
**Dependencies:** STEP_01

---

## Goal

Create searchable, sortable API parameter table using existing Table components.

---

## What We're Reusing

- ✅ `Table`, `TableHeader`, `TableBody`, `TableRow`, `TableCell`, `TableHead` from `@rbee/ui/atoms`
- ✅ `Badge` from `@rbee/ui/atoms`
- ✅ `Input` from `@rbee/ui/atoms`
- ✅ Icons from `lucide-react`

---

## Implementation

**File:** `/frontend/apps/user-docs/app/docs/_components/APIParameterTable.tsx`

```tsx
'use client'

import { useState } from 'react'
import { Table, TableHeader, TableBody, TableRow, TableCell, TableHead } from '@rbee/ui/atoms'
import { Badge } from '@rbee/ui/atoms'
import { Input } from '@rbee/ui/atoms'
import { Search } from 'lucide-react'

export interface APIParameter {
  name: string
  type: string
  required: boolean
  default?: string
  description: string
}

interface APIParameterTableProps {
  parameters: APIParameter[]
  searchable?: boolean
}

export function APIParameterTable({ 
  parameters, 
  searchable = true 
}: APIParameterTableProps) {
  const [search, setSearch] = useState('')
  
  const filtered = parameters.filter(p => 
    search === '' ||
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.description.toLowerCase().includes(search.toLowerCase()) ||
    p.type.toLowerCase().includes(search.toLowerCase())
  )
  
  return (
    <div className="my-6">
      {searchable && parameters.length > 5 && (
        <div className="mb-4 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search parameters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
      )}
      <div className="border rounded-lg overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[200px]">Parameter</TableHead>
              <TableHead className="w-[120px]">Type</TableHead>
              <TableHead className="w-[100px]">Required</TableHead>
              <TableHead className="w-[120px]">Default</TableHead>
              <TableHead>Description</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
                  No parameters found
                </TableCell>
              </TableRow>
            ) : (
              filtered.map(param => (
                <TableRow key={param.name}>
                  <TableCell className="font-mono font-semibold">
                    {param.name}
                  </TableCell>
                  <TableCell>
                    <code className="text-sm bg-muted px-2 py-1 rounded">
                      {param.type}
                    </code>
                  </TableCell>
                  <TableCell>
                    {param.required ? (
                      <Badge variant="destructive" className="text-xs">Required</Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">Optional</Badge>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-sm text-muted-foreground">
                    {param.default || '—'}
                  </TableCell>
                  <TableCell className="text-sm">
                    {param.description}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
      {filtered.length > 0 && filtered.length < parameters.length && (
        <p className="text-sm text-muted-foreground mt-2">
          Showing {filtered.length} of {parameters.length} parameters
        </p>
      )}
    </div>
  )
}
```

---

## Register in MDX

**File:** `/frontend/apps/user-docs/app/docs/mdx-components.tsx`

Add import:
```tsx
import { APIParameterTable } from './_components/APIParameterTable'
```

Add to exports:
```tsx
export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // ... existing exports ...
    
    // Custom wrappers
    Callout,
    CodeTabs,
    APIParameterTable,
    
    ...components,
  }
}
```

---

## Testing

**Test file:** `/app/docs/test-components.mdx`

```mdx
# API Parameter Table Test

<APIParameterTable
  parameters={[
    {
      name: 'model',
      type: 'string',
      required: true,
      description: 'Model ID to use for inference (e.g., "llama-3-8b")'
    },
    {
      name: 'prompt',
      type: 'string',
      required: true,
      description: 'Input text prompt for the model'
    },
    {
      name: 'temperature',
      type: 'float',
      required: false,
      default: '0.7',
      description: 'Sampling temperature (0.0-2.0). Higher values make output more random.'
    },
    {
      name: 'max_tokens',
      type: 'integer',
      required: false,
      default: '100',
      description: 'Maximum number of tokens to generate'
    },
    {
      name: 'top_p',
      type: 'float',
      required: false,
      default: '0.9',
      description: 'Nucleus sampling parameter (0.0-1.0)'
    },
    {
      name: 'stream',
      type: 'boolean',
      required: false,
      default: 'false',
      description: 'Enable streaming response via SSE'
    }
  ]}
/>
```

Run:
```bash
pnpm dev
# Visit http://localhost:7811/docs/test-components
# Test search functionality
# Test with <6 params (search should hide)
```

---

## Acceptance Criteria

- [ ] Table renders all parameters
- [ ] Search works (name, type, description)
- [ ] Search hides when <6 parameters
- [ ] Required/Optional badges display correctly
- [ ] Dark mode works
- [ ] Mobile responsive (horizontal scroll if needed)

---

## Next Step

→ **STEP_05_CREATE_REMOTE_HIVES_PAGE.md**
