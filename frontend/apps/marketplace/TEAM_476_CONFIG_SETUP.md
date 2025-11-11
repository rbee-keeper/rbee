# TEAM-476: Marketplace App Configuration Setup

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Add Tailwind, Biome, and TypeScript configuration matching commercial and user-docs apps

## Files Created/Modified

### 1. `package.json` - Complete Rewrite
**Changes:**
- ✅ Name: `@rbee/marketplace` (was `next`)
- ✅ Scripts: Added `lint`, `lint:fix`, `deploy`, `preview`, `typecheck`
- ✅ Port: `7823` (marketplace port)
- ✅ Dependencies: Added `@rbee/env-config`, `@rbee/ui`, Tailwind utilities
- ✅ DevDependencies: Added `@biomejs/biome`, `@repo/tailwind-config`, `postcss-nesting`
- ✅ Versions: Upgraded to Next.js 16.0.1, React 19.2.0

### 2. `biome.json` - Created
**Configuration:**
- ✅ VCS integration with Git
- ✅ Formatter: 120 char line width, 2-space indents
- ✅ Linter: Recommended rules enabled
- ✅ JavaScript: Single quotes, trailing commas, semicolons as needed
- ✅ Ignores: `.next/`, `.wrangler/`, `cloudflare-env.d.ts`, etc.

### 3. `tsconfig.json` - Updated
**Changes:**
- ✅ JSX: `react-jsx` (was `preserve`)
- ✅ Paths: `@/*` points to root (was `./src/*`)
- ✅ Types: Added `cloudflare-env.d.ts` and `node`
- ✅ Include: Added `.next/dev/types/**/*.ts`

### 4. `postcss.config.mjs` - Updated
**Changes:**
- ✅ Added `postcss-nesting` plugin
- ✅ Consistent format with commercial app

### 5. `src/app/globals.css` - Complete Rewrite
**Changes:**
- ✅ Imports: `@repo/tailwind-config/shared-styles.css`, `tw-animate-css`
- ✅ Source scanning: `../app/**/*.{ts,tsx}`, `../components/**/*.{ts,tsx}`
- ✅ Removed custom theme variables (now inherited from `@rbee/ui`)
- ✅ Documentation comments explaining architecture

## Configuration Alignment

### ✅ Matches Commercial App
- Biome configuration
- PostCSS configuration
- TypeScript configuration
- Package.json structure
- Global CSS imports

### ✅ Matches User-Docs App
- Tailwind v4 JIT scanning
- Shared styles import
- Component architecture

## Scripts Available

```bash
# Development
pnpm dev              # Start dev server on port 7823

# Building
pnpm build            # Build for production
pnpm start            # Start production server

# Linting
pnpm lint             # Check code with Biome
pnpm lint:fix         # Fix code with Biome

# Deployment
pnpm deploy           # Build and deploy to Cloudflare Pages
pnpm preview          # Build and preview locally

# Type Checking
pnpm typecheck        # Run TypeScript compiler check
pnpm cf-typegen       # Generate Cloudflare types
```

## Dependencies Added

### Production
- `@rbee/env-config` - Environment configuration
- `@rbee/ui` - Shared UI components
- `class-variance-authority` - CVA for component variants
- `clsx` - Conditional classnames
- `lucide-react` - Icons
- `next-themes` - Dark mode support
- `tailwind-merge` - Merge Tailwind classes
- `tailwindcss-animate` - Animation utilities
- `tw-animate-css` - Additional animations

### Development
- `@biomejs/biome` - Linter and formatter
- `@repo/tailwind-config` - Shared Tailwind config
- `postcss-nesting` - CSS nesting support

## Architecture

```
marketplace/
├── src/
│   └── app/
│       ├── globals.css          ← Imports @rbee/ui styles
│       ├── layout.tsx            ← Root layout
│       └── page.tsx              ← Home page
├── biome.json                    ← Linter/formatter config
├── tsconfig.json                 ← TypeScript config
├── postcss.config.mjs            ← PostCSS config
└── package.json                  ← Dependencies and scripts
```

## Next Steps

1. ✅ Install dependencies - DONE
2. ⏭️ Create app structure (pages, components)
3. ⏭️ Implement client-side fetchers (CivitAI, HuggingFace)
4. ⏭️ Add marketplace UI components from `@rbee/ui/marketplace`
5. ⏭️ Set up routing and navigation

## Verification

```bash
# Install dependencies
pnpm i

# Check linting
pnpm lint

# Check TypeScript
pnpm typecheck

# Start dev server
pnpm dev
```

---

**TEAM-476 RULE ZERO:** Configuration is now consistent across all frontend apps (commercial, user-docs, marketplace). All apps use Biome for linting, Tailwind v4 for styling, and share the same TypeScript configuration.
