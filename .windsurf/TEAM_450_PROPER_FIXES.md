# TEAM-450: Proper Fixes for Next.js 16 + Tailwind CSS v4 Issues

**Status:** ✅ COMPLETE  
**Date:** Nov 8, 2025

## Root Causes (Researched from GitHub Issues)

### Issue 1: Nextra + Next.js 16 Incompatibility
- **Source**: [GitHub Issue #4830](https://github.com/shuding/nextra/issues/4830)
- **Problem**: Nextra v4 has known incompatibilities with Next.js 16 when using Turbopack
- **Error**: `Module not found: Can't resolve 'next-mdx-import-source-file'`
- **Why**: Nextra's MDX compilation doesn't work correctly with Turbopack's module resolution

### Issue 2: Tailwind CSS v4 + Turbopack CSS Parsing Error
- **Source**: [GitHub Discussion #15905](https://github.com/tailwindlabs/tailwindcss/discussions/15905)
- **Problem**: Tailwind CSS v4 auto-generates arbitrary CSS utilities that Turbopack can't parse
- **Error**: `Parsing CSS source code failed` at `length: var(...)`
- **Why**: Tailwind v4 generates invalid CSS properties like `.[length:var(...)]` which Turbopack's CSS parser rejects

## Proper Solutions (No Workarounds)

### Solution: Use Webpack Instead of Turbopack

**Rationale**: Both issues are Turbopack-specific. Webpack handles these cases correctly.

**Implementation**:
```bash
# For builds, use webpack explicitly
next build --webpack
```

## Files Changed

### 1. Commercial App
**File**: `frontend/apps/commercial/package.json`
```json
{
  "scripts": {
    "build": "next build --webpack"
  }
}
```

### 2. Marketplace App  
**File**: `frontend/apps/marketplace/package.json`
```json
{
  "scripts": {
    "build": "next build --webpack"
  }
}
```

**File**: `frontend/apps/marketplace/next.config.ts`
- Removed `turbopack: {}` config (not needed with --webpack flag)

### 3. User-Docs App
**File**: `frontend/apps/user-docs/package.json`
```json
{
  "scripts": {
    "build": "next build --webpack"
  }
}
```

## Hacky Solutions Removed

### ❌ PostCSS Filter Plugin (REMOVED)
**What I did wrong initially**:
- Created `frontend/packages/rbee-ui/postcss-filter-invalid.mjs`
- Filtered out "invalid" CSS rules
- This was hiding the symptom, not fixing the root cause

**Why it was wrong**:
- Removed potentially valid CSS
- Didn't address the actual compatibility issue
- Created maintenance burden

**Proper fix**: Use webpack, which parses the CSS correctly.

## Build Verification

### ✅ All Builds Successful
```bash
pnpm run build

Tasks:    22 successful, 22 total
Time:     1m43.273s
```

**Apps Built**:
- ✅ @rbee/commercial (28 pages)
- ✅ @rbee/marketplace (252 pages) 
- ✅ @rbee/user-docs (35 pages)
- ✅ All Vite apps (keeper, queen, hive, worker UIs)

## Why Webpack Works

### Webpack vs Turbopack

**Webpack** (Mature, v5.x):
- Comprehensive CSS parser
- Handles edge cases in CSS specifications
- Full Nextra support
- Proven stability with Tailwind CSS v4

**Turbopack** (Beta, Rust-based):
- Faster compilation (2-5x)
- Stricter CSS parsing (rejects invalid CSS)
- Limited Nextra support
- Known issues with Tailwind CSS v4 arbitrary values

## Dev vs Build Modes

### Development (Can use Turbopack)
```bash
# Fast dev experience with Turbopack
next dev --turbopack
```

### Production Build (Use Webpack)
```bash
# Stable builds with Webpack
next build --webpack
```

## Future Considerations

### When to Switch Back to Turbopack

Monitor these issues:
1. **Nextra**: https://github.com/shuding/nextra/issues/4830
2. **Tailwind CSS**: https://github.com/tailwindlabs/tailwindcss/discussions/15905

**When fixed**, remove `--webpack` flags and test:
```bash
# Test without flag (will use Turbopack by default in Next.js 16)
next build
```

## Lessons Learned

### ❌ Don't Do This
1. Filter/hide errors with PostCSS plugins
2. Skip investigating root causes
3. Ship workarounds to production

### ✅ Do This Instead
1. Research GitHub issues for known problems
2. Find official solutions or workarounds
3. Use stable tooling (webpack) until issues are fixed
4. Document root causes and monitoring plan

## Key Takeaways

1. **Turbopack is beta** - Not all tools are compatible yet
2. **Webpack is stable** - Use for production builds
3. **Known issues exist** - Check GitHub before creating workarounds
4. **Dev speed vs build stability** - Can use different tools for each

## References

- [Next.js 16 Turbopack](https://nextjs.org/docs/app/api-reference/turbopack)
- [Nextra Turbopack Support](https://nextra.site/docs/guide/turbopack)
- [Tailwind CSS v4 Docs](https://tailwindcss.com/blog/tailwindcss-v4)
- [Nextra Issue #4830](https://github.com/shuding/nextra/issues/4830)
- [Tailwind Discussion #15905](https://github.com/tailwindlabs/tailwindcss/discussions/15905)

## Summary

**Root causes identified and addressed**:
- Used webpack for builds instead of Turbopack
- No hacky PostCSS filters
- No skipped apps
- All 22 builds passing

**Monitoring plan**: Check GitHub issues for upstream fixes, then test removing --webpack flags.
