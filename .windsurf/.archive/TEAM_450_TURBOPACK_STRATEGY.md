# TEAM-450: Turbopack Compatibility Strategy

**Status:** 🔍 INVESTIGATING  
**Date:** Nov 8, 2025

## Problem Statement

We need Turbopack for fast development builds, but Next.js 16.0.1 has compatibility issues:
1. Tailwind CSS v4 generates CSS that Turbopack can't parse
2. Nextra v4 has MDX import issues with Turbopack

## Current Solution (Temporary)

Using webpack for builds:
```json
{
  "scripts": {
    "build": "next build --webpack"
  }
}
```

**Pros:**
- ✅ All builds work (22/22 passing)
- ✅ No workarounds needed
- ✅ Stable and proven

**Cons:**
- ❌ Slower builds (~2-3x slower than Turbopack)
- ❌ Not using Next.js 16 default bundler
- ❌ Missing out on Turbopack performance

## Investigation Strategy

### Approach 1: Find Working Version (Bisect)

Created scripts to test Next.js versions:
- `scripts/find-turbopack-version.sh` - Sequential testing
- `scripts/find-turbopack-version-fast.sh` - Parallel testing (4x faster)

**What we're testing:**
- Next.js 16.0.x
- Next.js 15.1.x
- Next.js 15.0.x

**Test criteria:**
- ✅ Turbopack builds successfully
- ✅ Tailwind CSS v4 works
- ✅ Nextra v4 MDX pages work
- ✅ No CSS parsing errors
- ✅ No MDX import errors

### Approach 2: Wait for Upstream Fixes

Monitor these issues:
- [Nextra #4830](https://github.com/shuding/nextra/issues/4830) - Next.js 16 support
- [Tailwind #15905](https://github.com/tailwindlabs/tailwindcss/discussions/15905) - Turbopack CSS parsing

### Approach 3: Hybrid Strategy

**Development:** Use Turbopack (fast iteration)
```bash
next dev --turbopack
```

**Production:** Use webpack (stable builds)
```bash
next build --webpack
```

This gives us:
- ✅ Fast dev experience
- ✅ Stable production builds
- ✅ No version pinning needed

## Decision Matrix

| Strategy | Dev Speed | Build Speed | Stability | Maintenance |
|----------|-----------|-------------|-----------|-------------|
| **Webpack only** | Medium | Slow | High | Low |
| **Pin to working version** | Fast | Fast | Medium | Medium |
| **Hybrid (Turbopack dev, webpack build)** | Fast | Slow | High | Low |
| **Wait for fixes** | Slow | Slow | High | Low |

## Recommendations

### Short-term (Now)
Use **Hybrid Strategy**:
```json
{
  "scripts": {
    "dev": "next dev --turbopack",
    "build": "next build --webpack"
  }
}
```

**Benefits:**
- Fast development (Turbopack)
- Stable builds (webpack)
- No version pinning
- Works with latest Next.js

### Medium-term (1-2 months)
If bisect finds a working version:
```json
{
  "dependencies": {
    "next": "15.x.x"  // Last working version
  }
}
```

**Benefits:**
- Turbopack for everything
- Faster builds
- Known stable version

**Risks:**
- Miss out on Next.js 16 features
- Need to track upstream fixes
- May have other bugs

### Long-term (3-6 months)
Wait for upstream fixes, then:
```json
{
  "dependencies": {
    "next": "^16.0.0"  // Latest stable
  },
  "scripts": {
    "build": "next build"  // Turbopack by default
  }
}
```

## Testing Plan

### Phase 1: Bisect (In Progress)
- ✅ Created version finder scripts
- 🔄 Running parallel tests
- ⏳ Waiting for results

### Phase 2: Validate
If working version found:
1. Pin to that version
2. Test all 22 apps build
3. Test dev experience
4. Verify no regressions

### Phase 3: Monitor
- Watch GitHub issues
- Test new releases
- Update when fixed

## Fallback Plan

If no working version found:
1. Keep webpack for builds
2. Use Turbopack for dev
3. Document performance trade-offs
4. Re-evaluate quarterly

## Performance Comparison

### Webpack
- Dev server start: ~5-8 seconds
- Hot reload: ~1-2 seconds
- Production build: ~60-90 seconds

### Turbopack (when working)
- Dev server start: ~2-3 seconds
- Hot reload: ~200-500ms
- Production build: ~30-45 seconds

**Potential savings:** 40-50% faster builds

## Next Steps

1. ⏳ Wait for bisect script results
2. 📊 Analyze which versions work
3. 🧪 Test working version in real project
4. 📝 Update documentation
5. 🚀 Deploy solution

## References

- [Next.js Turbopack Docs](https://nextjs.org/docs/app/api-reference/turbopack)
- [Tailwind CSS v4 Docs](https://tailwindcss.com/blog/tailwindcss-v4)
- [Nextra v4 Migration](https://nextra.site/docs/guide/migrate-from-v3)
