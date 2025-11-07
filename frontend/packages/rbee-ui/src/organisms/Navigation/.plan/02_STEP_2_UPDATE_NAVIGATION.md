# Step 2: Update Navigation to Accept Config

**TEAM-460** | Phase 2 of 5

## Objective

Modify `Navigation.tsx` to accept a config object and use the sub-components we extracted.

## Changes to Navigation.tsx

### 1. Add NavigationConfig Interface

```typescript
import { DropdownMenu } from './DropdownMenu'
import { LinkGroup } from './LinkGroup'
import { NavigationActions } from './NavigationActions'
import type { NavigationConfig } from './types'

interface NavigationProps {
  config: NavigationConfig
}

export function Navigation({ config }: NavigationProps) {
  // ...
}
```

### 2. Replace Hardcoded Sections

**Before (lines 64-402):**
```tsx
<NavigationMenu viewport={false}>
  <NavigationMenuList className="gap-2">
    {/* Platform Dropdown - HARDCODED */}
    <NavigationMenuItem>
      <NavigationMenuTrigger>Platform</NavigationMenuTrigger>
      {/* ... hardcoded links ... */}
    </NavigationMenuItem>
    {/* ... more hardcoded dropdowns ... */}
  </NavigationMenuList>
</NavigationMenu>
```

**After:**
```tsx
<div className="hidden md:flex items-center justify-center gap-6 font-sans">
  {config.sections.map((section, index) => {
    if (section.type === 'dropdown') {
      return (
        <DropdownMenu
          key={index}
          title={section.title}
          links={section.links}
          cta={section.cta}
          width={section.width}
        />
      )
    }
    
    if (section.type === 'linkGroup') {
      return (
        <LinkGroup
          key={index}
          links={section.links}
        />
      )
    }
    
    return null
  })}
</div>
```

### 3. Replace Hardcoded Actions

**Before (lines 435-440):**
```tsx
<div className="hidden md:flex items-center gap-3 justify-self-end">
  {/* Hardcoded docs link */}
  {/* Hardcoded github link */}
  {/* Hardcoded CTA */}
</div>
```

**After:**
```tsx
<NavigationActions
  docs={config.actions.docs}
  github={config.actions.github}
  cta={config.actions.cta}
/>
```

### 4. Remove Mobile Menu Hardcoded Content

Replace hardcoded accordion items with mapped sections from config.

## Rules

- ✅ **REPLACE** hardcoded values with config mapping
- ✅ **USE** the sub-components we created
- ✅ **KEEP** all styling classes
- ✅ **KEEP** all accessibility attributes
- ❌ **DO NOT** change the overall structure
- ❌ **DO NOT** remove the mobile menu

## Backward Compatibility

**IMPORTANT:** Keep the old Navigation working until we update the commercial app.

Option 1: Default config
```typescript
export function Navigation({ config = defaultCommercialConfig }: NavigationProps) {
```

Option 2: Two exports
```typescript
export function Navigation({ config }: NavigationProps) { ... }
export function LegacyNavigation() { return <Navigation config={defaultCommercialConfig} /> }
```

## Verification

After this step:
- [ ] Navigation accepts config prop
- [ ] Dropdown sections render correctly
- [ ] LinkGroup sections render correctly
- [ ] Actions render correctly
- [ ] Mobile menu works
- [ ] No TypeScript errors
- [ ] Commercial site still works (with default config)

---

**Next:** `03_STEP_3_CREATE_CONFIGS.md`
