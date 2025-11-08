# Turbopack Version Finder Scripts

## Purpose

Find the last working version of Next.js that's compatible with:
- Turbopack (default bundler)
- Tailwind CSS v4
- Nextra v4

## Scripts

### 1. Sequential Version Finder (Thorough)
```bash
./scripts/find-turbopack-version.sh
```

**Features:**
- Tests versions one at a time
- Detailed logging
- ~5-10 minutes per version

**Use when:** You want detailed logs for debugging

### 2. Parallel Version Finder (Fast)
```bash
./scripts/find-turbopack-version-fast.sh
```

**Features:**
- Tests 4 versions in parallel
- Quick results (~2-3 minutes total)
- Summary output

**Use when:** You just want to find the working version quickly

## What They Test

Each script creates a minimal Next.js app with:
- ✅ Turbopack bundler
- ✅ Tailwind CSS v4 (`@import "tailwindcss"`)
- ✅ Nextra v4 with MDX pages
- ✅ TypeScript
- ✅ React 19

Then attempts to build with Turbopack and reports:
- ✅ **SUCCESS**: Version works
- ❌ **CSS_PARSE_ERROR**: Tailwind CSS v4 parsing issue
- ❌ **NEXTRA_ERROR**: Nextra MDX import issue
- ❌ **BUILD_FAILED**: Other build errors
- ❌ **INSTALL_FAILED**: Dependency resolution issues

## Tested Versions

Currently tests Next.js versions:
- 16.0.1 (current, known broken)
- 16.0.0
- 15.1.x series (15.1.6 down to 15.1.0)
- 15.0.x series (15.0.3 down to 15.0.0)

To test more versions, edit the `VERSIONS` array in either script.

## Example Output

```bash
🚀 Fast Turbopack Version Finder
Testing 13 versions with 4 parallel jobs

Running tests...

==========================================
Test Results:
==========================================
✓ 16.0.1: SUCCESS
✗ 16.0.0: Tailwind CSS v4 parsing error
✗ 15.1.6: Nextra MDX import error
✗ 15.1.5: Build failed
...
==========================================

✓ Last working version: Next.js 16.0.1

To pin to this version:

  pnpm update next@16.0.1 --recursive
```

## After Finding a Working Version

### Option 1: Pin to Working Version
```bash
# Update all apps to the working version
pnpm update next@<version> --recursive

# Verify
pnpm run build
```

### Option 2: Use Webpack (Current Solution)
If no working version is found, keep using webpack:
```json
{
  "scripts": {
    "build": "next build --webpack"
  }
}
```

## Troubleshooting

### Script hangs
- Kill with Ctrl+C
- Check `/tmp/turbopack-tests/` for stuck processes
- Reduce `MAX_PARALLEL` in fast script

### All versions fail
- Check test logs in `/tmp/turbopack-tests/test-<version>/build.log`
- Try older Next.js 14.x versions
- Consider using webpack for builds

### False positives
- Run sequential script for detailed logs
- Manually verify the version in your actual project

## Cleanup

Test artifacts are stored in `/tmp/`:
```bash
# Clean up test directories
rm -rf /tmp/turbopack-tests
rm -f /tmp/turbopack-test-results.log
rm -f /tmp/turbopack-results.txt
```

## Related Issues

Monitor these for upstream fixes:
- [Nextra #4830](https://github.com/shuding/nextra/issues/4830) - Next.js 16 compatibility
- [Tailwind #15905](https://github.com/tailwindlabs/tailwindcss/discussions/15905) - Turbopack CSS parsing
