# Step 5: Cleanup

**TEAM-460** | Phase 5 of 5

## Objective

Delete temporary files and ensure the codebase is clean.

## Files to Delete

### 1. SimpleNavigation (MISTAKE)

**File:** `frontend/packages/rbee-ui/src/organisms/Navigation/SimpleNavigation.tsx`

**Reason:** This was a lazy shortcut that violated RULE ZERO. Delete it.

```bash
rm frontend/packages/rbee-ui/src/organisms/Navigation/SimpleNavigation.tsx
```

### 2. Navigation Backup

**File:** `frontend/packages/rbee-ui/src/organisms/Navigation/Navigation.tsx.backup`

**Reason:** Temporary backup, no longer needed after verification.

```bash
rm frontend/packages/rbee-ui/src/organisms/Navigation/Navigation.tsx.backup
```

### 3. Update Exports

**File:** `frontend/packages/rbee-ui/src/organisms/Navigation/index.ts`

**Before:**
```typescript
export { Navigation } from './Navigation'
export { SimpleNavigation } from './SimpleNavigation'
export type { SimpleNavigationProps } from './SimpleNavigation'
```

**After:**
```typescript
export { Navigation } from './Navigation'
export { DropdownMenu } from './DropdownMenu'
export { LinkGroup } from './LinkGroup'
export { NavigationActions } from './NavigationActions'
export type * from './types'
```

## Final File Structure

```
frontend/packages/rbee-ui/src/organisms/Navigation/
├── .plan/
│   ├── 00_OVERVIEW.md
│   ├── 01_STEP_1_EXTRACT_SUBCOMPONENTS.md
│   ├── 02_STEP_2_UPDATE_NAVIGATION.md
│   ├── 03_STEP_3_CREATE_CONFIGS.md
│   ├── 04_STEP_4_UPDATE_APPS.md
│   └── 05_STEP_5_CLEANUP.md
├── types.ts
├── DropdownMenu.tsx
├── LinkGroup.tsx
├── NavigationActions.tsx
├── Navigation.tsx (updated)
├── Navigation.stories.tsx (update if needed)
└── index.ts (updated)
```

## Verification Checklist

### Code Quality
- [ ] No hardcoded values in Navigation.tsx
- [ ] All sub-components have proper TypeScript types
- [ ] No unused imports
- [ ] No console.log statements
- [ ] No TODO comments

### Functionality
- [ ] Commercial app works (all dropdowns)
- [ ] Marketplace app works (all links)
- [ ] Mobile menu works in both apps
- [ ] Theme toggle works
- [ ] All links navigate correctly
- [ ] Analytics events fire

### Architecture
- [ ] Single Navigation component
- [ ] Config-driven (no hardcoded values)
- [ ] Reusable sub-components
- [ ] Type-safe configs
- [ ] No duplicate code

### Documentation
- [ ] All .plan/ files present
- [ ] README updated (if needed)
- [ ] Storybook stories work (if applicable)

## Final Verification Commands

```bash
# Build both apps
cd frontend
pnpm build

# Type check
pnpm typecheck

# Lint
pnpm lint

# Test (if tests exist)
pnpm test
```

## Success Criteria

✅ **Navigation component:**
- Accepts config prop
- No hardcoded values
- Works for both commercial and marketplace

✅ **Commercial app:**
- Uses commercialNavConfig
- All dropdowns work
- Analytics tracking works

✅ **Marketplace app:**
- Uses marketplaceNavConfig
- All links work
- 81 lines → 7 lines

✅ **Codebase:**
- No duplicate navigation code
- No temporary files
- Clean exports
- Type-safe

## Lines of Code Saved

**Before:**
- Navigation.tsx: 705 lines (with hardcoded values)
- MarketplaceNav.tsx: 81 lines (hand-rolled)
- **Total: 786 lines**

**After:**
- Navigation.tsx: ~200 lines (config-driven)
- DropdownMenu.tsx: ~100 lines
- LinkGroup.tsx: ~50 lines
- NavigationActions.tsx: ~80 lines
- types.ts: ~50 lines
- commercialNavConfig.ts: ~120 lines
- marketplaceNavConfig.ts: ~40 lines
- CommercialNav.tsx: 7 lines
- MarketplaceNav.tsx: 7 lines
- **Total: ~654 lines**

**Savings: 132 lines + NO DUPLICATION**

---

**DONE - Navigation refactor complete!**
