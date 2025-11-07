# Step 1: Extract Sub-Components (Atomic Design)

**TEAM-460** | Phase 1 of 5

## Objective

Extract reusable **MOLECULES** from the existing `Navigation.tsx` organism WITHOUT deleting anything.

**Atomic Design Pattern:**
- **Atoms:** Button, IconButton, ThemeToggle (already exist)
- **Molecules:** DropdownMenu, LinkGroup, NavigationActions (extract these)
- **Organism:** Navigation (uses molecules)

## What to Extract

### 1. DropdownMenu Component (MOLECULE)

**Location:** `frontend/packages/rbee-ui/src/molecules/DropdownMenu/DropdownMenu.tsx`

**Purpose:** Render a dropdown menu with links (Platform, Solutions, Industries style)

**Props:**
```typescript
interface DropdownMenuProps {
  title: string
  links: Array<{
    label: string
    href: string
    description?: string
    icon?: LucideIcon
  }>
  cta?: {
    label: string
    href?: string
    onClick?: () => void
  }
  width?: 'sm' | 'md' | 'lg'  // 280px, 560px, etc.
}
```

**Extract from:** Lines 68-428 in Navigation.tsx (the NavigationMenuItem blocks)

### 2. LinkGroup Component (MOLECULE)

**Location:** `frontend/packages/rbee-ui/src/molecules/LinkGroup/LinkGroup.tsx`

**Purpose:** Render a simple group of links (Models, Workers style)

**Props:**
```typescript
interface LinkGroupProps {
  links: Array<{
    label: string
    href: string
    badge?: string  // "Soon"
    disabled?: boolean
  }>
}
```

**This is NEW** - marketplace needs this pattern

### 3. NavigationActions Component (MOLECULE)

**Location:** `frontend/packages/rbee-ui/src/molecules/NavigationActions/NavigationActions.tsx`

**Purpose:** Render right-side actions (Docs, GitHub, Theme, CTA)

**Props:**
```typescript
interface NavigationActionsProps {
  docs?: {
    url: string
    label?: string
  }
  github?: {
    url: string
  }
  cta?: {
    label: string
    href?: string
    onClick?: () => void
    ariaLabel?: string
  }
}
```

**Extract from:** Lines 435-440 in Navigation.tsx (Zone C)

## Implementation Order

1. Create `types.ts` in Navigation organism (shared types)
2. Create `molecules/DropdownMenu/` directory with:
   - `DropdownMenu.tsx` - Copy dropdown logic from Navigation
   - `index.ts` - Export
3. Create `molecules/LinkGroup/` directory with:
   - `LinkGroup.tsx` - New component for marketplace
   - `index.ts` - Export
4. Create `molecules/NavigationActions/` directory with:
   - `NavigationActions.tsx` - Copy actions from Navigation
   - `index.ts` - Export
5. Update `molecules/index.ts` to export all three
6. Test each molecule in isolation

## Rules

- ❌ **DO NOT DELETE** anything from Navigation.tsx yet
- ❌ **DO NOT MODIFY** Navigation.tsx yet
- ✅ **COPY** code from Navigation.tsx to new files
- ✅ **ADD** proper TypeScript types
- ✅ **KEEP** all existing styling and behavior

## Files to Create

```
frontend/packages/rbee-ui/src/
├── molecules/
│   ├── DropdownMenu/
│   │   ├── DropdownMenu.tsx (new)
│   │   └── index.ts (new)
│   ├── LinkGroup/
│   │   ├── LinkGroup.tsx (new)
│   │   └── index.ts (new)
│   ├── NavigationActions/
│   │   ├── NavigationActions.tsx (new)
│   │   └── index.ts (new)
│   └── index.ts (update - export new molecules)
└── organisms/
    └── Navigation/
        ├── types.ts (new - shared types)
        └── Navigation.tsx (unchanged)
```

## Verification

After this step:
- [ ] DropdownMenu renders correctly with test data
- [ ] LinkGroup renders correctly with test data
- [ ] NavigationActions renders correctly with test data
- [ ] Navigation.tsx still works (unchanged)
- [ ] No TypeScript errors

---

**Next:** `02_STEP_2_UPDATE_NAVIGATION.md`
