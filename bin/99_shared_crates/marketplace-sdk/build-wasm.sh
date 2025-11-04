#!/bin/bash
# TEAM-402: Build WASM for all targets

set -e

echo "🔨 Building WASM for bundler target..."
wasm-pack build --target bundler --out-dir pkg/bundler

echo "🔨 Building WASM for web target..."
wasm-pack build --target web --out-dir pkg/web

echo "✅ WASM built for all targets"
echo ""
echo "📦 Output:"
echo "  - pkg/bundler/marketplace_sdk.js"
echo "  - pkg/bundler/marketplace_sdk.d.ts (TypeScript types)"
echo "  - pkg/bundler/marketplace_sdk_bg.wasm"
