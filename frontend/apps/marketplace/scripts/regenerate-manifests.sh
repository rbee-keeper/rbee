#!/usr/bin/env bash
# TEAM-467: Regenerate model manifests using WASM SDK
# FAIL FAST - Exit immediately on ANY error, show live output

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "🔄 Regenerating model manifests..."
echo ""

# Check if we're in production mode
if [ "${NODE_ENV:-}" != "production" ]; then
  echo "⚠️  Setting NODE_ENV=production (required for manifest generation)"
  export NODE_ENV=production
fi

# TEAM-467: FAIL FAST - Run directly with live output, exit on error
echo "📦 Running manifest generation (live output)..."
echo ""
pnpm run generate:manifests

echo ""
echo "✅ Manifests regenerated successfully!"
echo ""
echo "📊 Manifest files:"
ls -lh public/manifests/ | grep -E "(models\\.json|hf-filter|civitai-filter)" | head -10
echo ""
echo "💡 To use in dev: Restart your dev server to pick up new manifests"
echo "💡 To deploy: Run 'pnpm run build' to include in production build"
