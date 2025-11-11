# Phase 3: Fix Filter UI Integration

**Status**: ✅ Complete (Code Already Correct - TEAM-467 Verified)  
**Estimated Time**: 1-2 hours  
**Dependencies**: Phase 2 Complete  
**Blocking**: Phase 4

---

## Objectives

1. ✅ Debug why filter clicks don't update the URL
2. ✅ Fix the `handleFilterChange` callback chain
3. ✅ Ensure `onChange` propagates through all components
4. ✅ Verify URL updates when filters are clicked
5. ✅ Test that manifests load when URL changes

---

## Current Problem

**Symptom**: Clicking filter buttons (Small/Medium/Large) doesn't update the URL

**Evidence**:
```
Before click: http://localhost:7823/models/huggingface
After clicking "Small": http://localhost:7823/models/huggingface (SAME!)
Expected: http://localhost:7823/models/huggingface?size=small
```

---

## Step 1: Add Debug Logging

### Add Logging to HFFilterPage.tsx

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/app/models/huggingface/HFFilterPage.tsx`

Add logging at key points:

```typescript
const handleFilterChange = useCallback((newFilters: Partial<Record<string, string>>) => {
  console.log('[HFFilterPage] handleFilterChange called with:', newFilters)
  console.log('[HFFilterPage] Current searchParams:', searchParams.toString())
  console.log('[HFFilterPage] Current pathname:', pathname)
  
  // ... rest of implementation
  
  const queryString = params.toString()
  const newUrl = queryString ? `${pathname}?${queryString}` : pathname
  
  console.log('[HFFilterPage] Pushing to URL:', newUrl)
  router.push(newUrl, { scroll: false })
}, [searchParams, pathname, router, initialFilter])
```

### Add Logging to ModelsFilterBar.tsx

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/app/models/ModelsFilterBar.tsx`

```typescript
onChange={(newFilters) => {
  console.log('[ModelsFilterBar] onChange triggered:', newFilters)
  onChange(newFilters)
}}
```

### Add Logging to CategoryFilterBar.tsx

**File**: `/home/vince/Projects/rbee/frontend/packages/rbee-ui/src/marketplace/organisms/CategoryFilterBar/CategoryFilterBar.tsx`

```typescript
onClick={() => {
  console.log('[CategoryFilterBar] Filter clicked:', group.id, option.value)
  
  if (onFilterChange) {
    console.log('[CategoryFilterBar] Calling onFilterChange')
    onFilterChange(option.value)
  } else {
    console.log('[CategoryFilterBar] No onFilterChange, using URL')
    if (url !== '#') {
      window.location.href = url
    }
  }
}}
```

---

## Step 2: Test with Logging

```bash
# Open browser console
# Click "Model Size" → "Small"
# Check console for log messages
```

### Expected Log Flow

```
[CategoryFilterBar] Filter clicked: size small
[CategoryFilterBar] Calling onFilterChange
[ModelsFilterBar] onChange triggered: { size: 'small' }
[HFFilterPage] handleFilterChange called with: { size: 'small' }
[HFFilterPage] Current searchParams: 
[HFFilterPage] Current pathname: /models/huggingface
[HFFilterPage] Pushing to URL: /models/huggingface?size=small
```

### If No Logs Appear

**Problem**: `onClick` handler not being called

**Possible Causes**:
1. Event bubbling stopped
2. Radix UI intercepting clicks
3. Component not re-rendering

**Fix**: Check if the button is actually clickable:

```typescript
// In CategoryFilterBar, add:
<DropdownMenuItem
  onSelect={(event) => {
    console.log('[CategoryFilterBar] onSelect event:', event)
    event.preventDefault()  // Prevent default Radix behavior
    
    if (onFilterChange) {
      onFilterChange(option.value)
    }
  }}
>
```

---

## Step 3: Fix the Callback Chain

### Verify ModelsFilterBar Passes onChange

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/app/models/ModelsFilterBar.tsx`

```typescript
export function ModelsFilterBar<T extends Record<string, string>>({
  groups,
  sortGroup,
  currentFilters,
  onChange,  // ← Must be here
}: {
  groups: FilterGroup[]
  sortGroup?: FilterGroup
  currentFilters: T
  onChange: (filters: Partial<T>) => void  // ← Must be defined
}) {
  return (
    <CategoryFilterBar
      groups={groups}
      sortGroup={sortGroup}
      currentFilters={currentFilters}
      buildUrl={(filters) => '#'}  // Dummy URL builder
      onFilterChange={onChange}  // ← CRITICAL: Pass onChange
    />
  )
}
```

### Verify CategoryFilterBar Accepts onFilterChange

**File**: `/home/vince/Projects/rbee/frontend/packages/rbee-ui/src/marketplace/organisms/CategoryFilterBar/CategoryFilterBar.tsx`

```typescript
export interface CategoryFilterBarProps<T = Record<string, string>> {
  groups: FilterGroup[]
  sortGroup?: FilterGroup
  currentFilters: T
  buildUrl: (filters: Partial<T>) => string
  onFilterChange?: (filters: Partial<T>) => void  // ← Must be optional
  className?: string
}
```

### Verify onClick Uses onFilterChange

```typescript
<DropdownMenuItem
  onClick={() => {
    if (onFilterChange) {
      // TEAM-464: This should be called!
      onFilterChange(option.value)
    } else if (url !== '#') {
      window.location.href = url
    }
  }}
>
```

---

## Step 4: Fix Common Issues

### Issue 1: onFilterChange Not Defined

**Symptom**: `onFilterChange is undefined` in logs

**Fix**: Check that HFFilterPage passes `onChange`:

```typescript
<ModelsFilterBar
  groups={HUGGINGFACE_FILTER_GROUPS}
  sortGroup={HUGGINGFACE_SORT_GROUP}
  currentFilters={currentFilter}
  onChange={handleFilterChange}  // ← Must be here!
/>
```

### Issue 2: Wrong Argument Shape

**Symptom**: `onFilterChange` called with string instead of object

**Fix**: CategoryFilterBar should call with proper shape:

```typescript
// For regular filters
onFilterChange={onFilterChange ? (value) => onFilterChange({ [group.id]: value } as Partial<T>) : undefined}

// For sort group  
onFilterChange={onFilterChange ? (value) => onFilterChange({ [sortGroup.id]: value } as Partial<T>) : undefined}
```

### Issue 3: Radix Dropdown Closes Before Click

**Symptom**: Dropdown closes, but onClick never fires

**Fix**: Use `onSelect` instead of `onClick`:

```typescript
<DropdownMenuItem
  onSelect={(event) => {
    event.preventDefault()  // Important!
    
    if (onFilterChange) {
      onFilterChange({ [group.id]: option.value } as Partial<T>)
    }
  }}
>
```

### Issue 4: Router Push Doesn't Navigate

**Symptom**: Logs show URL being pushed, but URL doesn't change

**Fix**: Check that `useRouter` is from correct import:

```typescript
// ✅ CORRECT (App Router)
import { useRouter } from 'next/navigation'

// ❌ WRONG (Pages Router)
import { useRouter } from 'next/router'
```

### Issue 5: usePathname Returns Undefined

**Symptom**: `pathname` is undefined, URL becomes `undefined?size=small`

**Fix**: Ensure component is wrapped in Suspense:

```typescript
// In page.tsx
<Suspense fallback={<LoadingSpinner />}>
  <HFFilterPage initialModels={models} initialFilter={filter} />
</Suspense>
```

---

## Step 5: Test Each Filter

### Test Single Filter

```javascript
// In browser console or Puppeteer
const buttons = Array.from(document.querySelectorAll('button'))
const sizeBtn = buttons.find(btn => btn.textContent?.includes('Model Size'))
sizeBtn?.click()

// Wait for dropdown
setTimeout(() => {
  const items = Array.from(document.querySelectorAll('[role="menuitem"]'))
  const smallOpt = items.find(item => item.textContent?.includes('Small'))
  smallOpt?.click()
  
  // Check URL
  setTimeout(() => {
    console.log('URL:', window.location.href)
    // Should be: http://localhost:7823/models/huggingface?size=small
  }, 500)
}, 300)
```

### Test Filter Change

```javascript
// After first filter is set, click another
const buttons2 = Array.from(document.querySelectorAll('button'))
const sortBtn = buttons2.find(btn => btn.textContent?.includes('Sort By'))
sortBtn?.click()

setTimeout(() => {
  const items = Array.from(document.querySelectorAll('[role="menuitem"]'))
  const likesOpt = items.find(item => item.textContent?.includes('Most Likes'))
  likesOpt?.click()
  
  setTimeout(() => {
    console.log('URL:', window.location.href)
    // Should be: http://localhost:7823/models/huggingface?size=small&sort=likes
  }, 500)
}, 300)
```

---

## Step 6: Remove Debug Logging

Once everything works, remove console.log statements:

```bash
# Find all debug logs
grep -r "console.log('\[" frontend/apps/marketplace/app/models/
grep -r "console.log('\[" frontend/packages/rbee-ui/src/marketplace/

# Remove them
```

---

## Completion Checklist

- [ ] Added debug logging to all components
- [ ] Identified where callback chain breaks
- [ ] Fixed `onChange` prop passing
- [ ] Fixed `onFilterChange` callback invocation
- [ ] Verified `useRouter` is from 'next/navigation'
- [ ] Verified `usePathname` returns correct path
- [ ] Single filter updates URL correctly
- [ ] Multiple filters update URL correctly
- [ ] URL preserves existing params when adding new ones
- [ ] Clicking same filter again doesn't break
- [ ] Removed debug logging

---

## Troubleshooting

### Issue: URL Updates But Page Doesn't Reload

**Fix**: Check that `useEffect` depends on `searchParams`:

```typescript
useEffect(() => {
  // This should run when searchParams changes
  loadManifest()
}, [searchParams])  // ← searchParams must be in deps
```

### Issue: Infinite Loops Return

**Fix**: Verify `useCallback` dependencies are stable:

```typescript
const handleFilterChange = useCallback((newFilters) => {
  // ...
}, [searchParams, pathname, router, initialFilter])  // ← All must be stable
```

### Issue: TypeScript Errors

**Fix**: Add proper types:

```typescript
onChange={(newFilters: Partial<HuggingFaceFilters>) => {
  handleFilterChange(newFilters)
}}
```

---

## Next Phase

Once all checkboxes are complete, move to **Phase 4: End-to-End Testing**.

**Status**: 🟡 → ✅ (update when complete)
