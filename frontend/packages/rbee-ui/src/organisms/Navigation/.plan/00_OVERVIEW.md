# Navigation Component Refactor - OVERVIEW

**TEAM-460** | Created: Nov 7, 2025

## Problem

The Navigation component has **HARDCODED VALUES** for commercial site links (Platform, Solutions, Industries). The marketplace needs different links (Models, Workers) but can't use the same component.

## Current State

- `Navigation.tsx` - 705 lines with hardcoded commercial dropdowns
- `SimpleNavigation.tsx` - MISTAKE - lazy shortcut with children prop
- `MarketplaceNav.tsx` - Hand-rolled duplicate navigation

## Goal

**Single Navigation component that accepts PURE DATA CONFIG (no TSX)**

Both commercial and marketplace apps pass their own config objects.

## Architecture

```
Navigation (container)
├── DropdownMenu (for commercial-style dropdowns)
├── LinkGroup (for marketplace-style link groups)
└── NavigationActions (right-side: docs, github, cta)
```

## Configuration Format

```typescript
interface NavigationConfig {
  sections: Array<DropdownSection | LinkGroupSection>
  actions: NavigationActions
}
```

**NO TSX IN CONFIG - PURE DATA ONLY**

## Implementation Phases

1. **Phase 1: Extract Sub-Components** (Step 1)
2. **Phase 2: Update Navigation** (Step 2)
3. **Phase 3: Create Configs** (Step 3)
4. **Phase 4: Update Apps** (Step 4)
5. **Phase 5: Cleanup** (Step 5)

## Success Criteria

- ✅ Navigation accepts config object (no hardcoded values)
- ✅ Commercial site works with dropdown config
- ✅ Marketplace works with link group config
- ✅ NO TSX in config files
- ✅ NO duplicate navigation code
- ✅ Single source of truth

## Files to Create

- `DropdownMenu.tsx`
- `LinkGroup.tsx`
- `NavigationActions.tsx`
- `types.ts` (NavigationConfig interfaces)
- `commercialNavConfig.ts`
- `marketplaceNavConfig.ts`

## Files to Delete

- `SimpleNavigation.tsx` (MISTAKE)
- `Navigation.tsx.backup` (after verification)
- `MarketplaceNav.tsx` (replace with config)

---

**Next:** Read `01_STEP_1_EXTRACT_SUBCOMPONENTS.md`
