# Component-Level Changes

**Date:** November 3, 2025  
**Goal:** Specific code changes needed for components and templates

---

## Color Token Updates

### File: `packages/rbee-ui/src/tokens/theme-tokens.css`

#### Light Mode - Warm Colors
```css
:root {
  --background: #fdfbf7;           /* warm cream */
  --card: #fffef9;                 /* warm white */
  --primary: #e6a23c;              /* honey gold */
  --accent: #f0b454;               /* lighter honey */
  --secondary: #f5f3ed;            /* warm beige */
  --muted: #f8f6f0;
  --border: #e8e3d8;               /* warm tan */
}
```

#### Dark Mode - CRITICAL FIX (WCAG Compliance)
```css
.dark {
  --background: #1a1612;           /* warm dark gray */
  --card: #252118;
  --primary: #f0b454;              /* bright honey - PASSES WCAG */
  --accent: #f5c675;
  --muted: #2a2520;
  --border: #3a342d;
}
```

---

## New Components

### 1. ComparisonCard Molecule
**File:** `packages/rbee-ui/src/molecules/ComparisonCard/ComparisonCard.tsx`

Shows "Without rbee" vs "With rbee" comparison with visual contrast.

### 2. StatsBar Molecule
**File:** `packages/rbee-ui/src/molecules/StatsBar/StatsBar.tsx`

Displays social proof numbers (5,000+ users, 15,000+ stars, etc.)

### 3. ComparisonCardTemplate
**File:** `packages/rbee-ui/src/templates/ComparisonCardTemplate/`

Template wrapper for ComparisonCard with TemplateContainer.

---

## Component Updates

### Badge - Add Pulse Animation
```typescript
interface BadgeProps {
  showPulse?: boolean
}
```

### HeroTemplate - Add Social Proof
```typescript
interface HeroTemplateProps {
  socialProof?: {
    text?: string
    items: Array<{ icon: string; label: string }>
  }
}
```

### UseCasesTemplate - Add Concrete Examples
```typescript
items: Array<{
  concreteExample?: string  // NEW
  // ... existing fields
}>
```

### HowItWorks - Add Descriptions
```typescript
steps: Array<{
  description?: string  // NEW
  details?: string[]    // NEW
  // ... existing fields
}>
```

---

See full implementation details in separate files.
