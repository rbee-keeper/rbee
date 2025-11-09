/**
 * Unified focus ring utility for consistent focus states across all interactive atoms.
 *
 * Uses CSS variables from theme-tokens.css:
 * - --focus-ring-color
 * - --focus-ring-width
 * - --focus-ring-offset
 * - --bg-canvas (for ring offset color)
 *
 * Apply to: Button, Input, Textarea, Select, Checkbox, Radio, Switch,
 *           Tabs.Trigger, MenuItem, Link, and any focusable element.
 */

export const focusRing =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background'

/**
 * Variant for elements that need tighter focus (e.g., small icons, compact controls)
 */
export const focusRingTight =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background'

/**
 * Variant for destructive actions (uses destructive color instead of brand)
 */
export const focusRingDestructive =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-destructive focus-visible:ring-offset-2 focus-visible:ring-offset-background'

/**
 * Brand-consistent link styling with amber underline and focus states
 * Apply to: Button link variant, NavLink, inline CTAs
 * Dark mode: default text-[color:var(--accent)], hover text-white with decoration-amber-400
 * Visited: uses brand 700 (#b45309) to communicate state
 *
 * TEAM-450: Removed typed arbitrary values ([length:...], [color:...])
 * because they cause Turbopack CSS parsing errors. Using untyped arbitrary values instead.
 */
export const brandLink =
  'text-[var(--accent)] underline underline-offset-2 decoration-amber-400 ' +
  'hover:text-white hover:decoration-amber-300 ' +
  'visited:text-[#b45309] ' +
  'focus-visible:outline-none ' +
  'focus-visible:ring-[var(--focus-ring-width)] ' +
  'focus-visible:ring-[var(--ring)] ' +
  'focus-visible:ring-offset-[var(--focus-ring-offset)] ' +
  'focus-visible:ring-offset-[var(--background)] ' +
  'transition-colors'

/**
 * Inverse-contrast focus for amber surfaces (opt-in)
 * Use when element bg ~ amber (e.g., Badge accent variant in dark)
 * Swaps ring to white to prevent blending with amber fills
 *
 * TEAM-450: Removed typed arbitrary value ([color:...]) to fix Turbopack CSS parsing errors
 */
export const focusInverse =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--background)]'
