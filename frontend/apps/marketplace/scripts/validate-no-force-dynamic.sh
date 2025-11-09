#!/usr/bin/env bash
# TEAM-462: Build validation - FORBID force-dynamic
# This script FAILS the build if any page uses force-dynamic
# force-dynamic causes Cloudflare Worker CPU timeouts (Error 1102)

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⛔ VALIDATING: No force-dynamic in marketplace pages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Search for force-dynamic in app directory
# Exclude robots.ts, sitemap.ts, opengraph-image.tsx (force-static is OK)
# Match actual export statements, not comments
FORBIDDEN_FILES=$(grep -r "^export const dynamic = ['\"]force-dynamic['\"]" app/ \
  --include="*.tsx" \
  --include="*.ts" \
  --exclude="robots.ts" \
  --exclude="sitemap.ts" \
  --exclude="opengraph-image.tsx" \
  || true)

if [ -n "$FORBIDDEN_FILES" ]; then
  echo ""
  echo "❌ BUILD VALIDATION FAILED"
  echo ""
  echo "Found 'export const dynamic = \"force-dynamic\"' in the following files:"
  echo ""
  echo "$FORBIDDEN_FILES"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "⛔ force-dynamic is FORBIDDEN in marketplace pages"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "WHY THIS IS FORBIDDEN:"
  echo "  • force-dynamic causes Server-Side Rendering on EVERY request"
  echo "  • SSR runs in Cloudflare Workers with strict CPU limits (50-200ms)"
  echo "  • External API calls (HuggingFace, CivitAI) exceed these limits"
  echo "  • Result: Error 1102 'Worker exceeded resource limits'"
  echo ""
  echo "HOW TO FIX:"
  echo "  1. Remove 'export const dynamic = \"force-dynamic\"'"
  echo "  2. Use Static Site Generation (SSG) with generateStaticParams()"
  echo "  3. Pre-render pages at build time (locally)"
  echo "  4. Deploy static HTML to Cloudflare"
  echo ""
  echo "IF BUILD FAILS:"
  echo "  • Fix the API integration (don't add force-dynamic)"
  echo "  • Reduce filter combinations to working ones"
  echo "  • Handle API errors gracefully (return empty array)"
  echo "  • Test each filter combination individually"
  echo ""
  echo "NEVER USE force-dynamic AS A 'QUICK FIX'"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi

echo ""
echo "✅ VALIDATION PASSED: No force-dynamic found"
echo "✅ All pages will be statically generated at build time"
echo ""
