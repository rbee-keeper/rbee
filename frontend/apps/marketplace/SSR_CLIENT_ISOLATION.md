# SSR/Client Component Isolation Fix

**Date:** Nov 4, 2025  
**Issue:** `useState` hook contaminating SSR build  
**Root Cause:** Client-side components exported from main package index

## Problem

The marketplace package was exporting **everything** from its main index:
```typescript
// ❌ WRONG - Exports hooks and templates that use hooks
export * from './hooks'  // useModelFilters (uses useState)
export * from './templates/ModelListTableTemplate'  // uses useModelFilters
export * from './organisms/FilterBar'  // uses controlled state
```

When Next.js analyzed imports:
```typescript
import { ModelTable } from '@rbee/ui/marketplace'
```

It would also analyze ALL exports from that package, including the hooks, causing the build error even though we weren't using them.

## Solution

**Split exports into SSR-safe and client-only:**

### SSR-Safe Exports (Main Index)
```typescript
// ✅ Safe to import in Server Components
export * from './organisms/ModelTable'  // Pure presentation, no hooks
export * from './organisms/ModelCard'
export * from './molecules/ModelMetadataCard'
// ... etc
```

### Client-Only Exports (Explicit Imports)
```typescript
// ❌ NOT exported from main index
// Must import explicitly in 'use client' components:
// import { useModelFilters } from '@rbee/ui/marketplace/hooks/useModelFilters'
// import { ModelListTableTemplate } from '@rbee/ui/marketplace/templates/ModelListTableTemplate'
```

## Files Changed

### 1. `/packages/rbee-ui/src/marketplace/index.ts`

**Removed from exports:**
- `export * from './hooks'` - All hooks (useModelFilters, etc.)
- `export * from './templates/ModelListTableTemplate'` - Uses hooks
- `export * from './organisms/FilterBar'` - Uses controlled state

**Added documentation:**
```typescript
// ============================================================================
// CLIENT-ONLY EXPORTS (Use hooks, require 'use client')
// ============================================================================
// Import these explicitly in client components only:
// - ./organisms/FilterBar (uses controlled state)
// - ./hooks/useModelFilters (React hooks)
// - ./templates/ModelListTableTemplate (uses useModelFilters)
```

### 2. `/apps/marketplace/app/models/page.tsx`

**Only imports SSR-safe components:**
```typescript
import { ModelTable } from '@rbee/ui/marketplace'  // ✅ Pure presentation
import type { ModelTableItem } from '@rbee/ui/marketplace'  // ✅ Type only
```

**No client components imported** - Pure Server Component

## Architecture

```
@rbee/ui/marketplace (Main Index)
├─ SSR-Safe Exports ✅
│  ├─ ModelTable (pure presentation)
│  ├─ ModelCard (pure presentation)
│  └─ ... (no hooks)
│
└─ Client-Only (Not exported) ⚠️
   ├─ hooks/useModelFilters (uses useState)
   ├─ templates/ModelListTableTemplate (uses hooks)
   └─ organisms/FilterBar (controlled state)
```

## Usage Patterns

### ✅ Server Component (SSR)
```typescript
// app/models/page.tsx
import { ModelTable } from '@rbee/ui/marketplace'

export default async function ModelsPage() {
  const models = await fetchModels()
  return <ModelTable models={models} />
}
```

### ✅ Client Component (CSR)
```typescript
// components/InteractiveModelList.tsx
'use client'

import { ModelListTableTemplate } from '@rbee/ui/marketplace/templates/ModelListTableTemplate'
import { useModelFilters } from '@rbee/ui/marketplace/hooks/useModelFilters'

export function InteractiveModelList({ models }) {
  return <ModelListTableTemplate models={models} />
}
```

## Benefits

✅ **Clean separation** - SSR and CSR components isolated  
✅ **No build errors** - Hooks don't contaminate SSR  
✅ **Explicit imports** - Clear which components need 'use client'  
✅ **Tree-shaking** - Unused client code not bundled in SSR  
✅ **Type safety** - TypeScript still works for both  

## Testing

**Build should succeed:**
```bash
cd frontend/apps/marketplace
rm -rf .next
pnpm build
```

**Expected output:**
```
✓ Compiled successfully
✓ Generating static pages (100/100)
✓ Finalizing page optimization
```

**No errors about:**
- `useState` in Server Components
- Hooks in SSR context
- Client-side code in static generation

## Key Takeaway

**Package exports matter for Next.js SSR!**

Even if you don't import a component, if it's exported from the main index, Next.js will analyze it and fail if it contains hooks.

**Solution:** Only export SSR-safe components from main index. Client-only components must be imported explicitly with full paths.
