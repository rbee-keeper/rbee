#!/bin/bash
# TEAM-457: Generate TypeScript types for all Cloudflare projects
# Run this after modifying any wrangler.jsonc file

set -e

echo "🔧 Generating Cloudflare TypeScript types..."
echo ""

# Next.js projects
echo "📦 Commercial frontend..."
cd frontend/apps/commercial
pnpm dlx wrangler types --silent
echo "✅ Types generated: frontend/apps/commercial/worker-configuration.d.ts"
cd ../../..

echo ""
echo "📦 Marketplace frontend..."
cd frontend/apps/marketplace
pnpm dlx wrangler types --silent
echo "✅ Types generated: frontend/apps/marketplace/worker-configuration.d.ts"
cd ../../..

echo ""
echo "📦 User docs frontend..."
cd frontend/apps/user-docs
pnpm dlx wrangler types --silent
echo "✅ Types generated: frontend/apps/user-docs/worker-configuration.d.ts"
cd ../../..

# Hono worker
echo ""
echo "📦 Hono worker catalog..."
cd bin/80-hono-worker-catalog
pnpm dlx wrangler types --silent
echo "✅ Types generated: bin/80-hono-worker-catalog/worker-configuration.d.ts"
cd ../..

echo ""
echo "✨ All Cloudflare types generated successfully!"
echo ""
echo "💡 Tip: Run this script after modifying any wrangler.jsonc file"
