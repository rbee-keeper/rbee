#!/bin/bash
# Migration script for moving pages from rbee-ui to commercial submodule
# This script helps migrate the CMS (PageProps) to the private repo

set -e  # Exit on error

echo "🚀 Starting pages migration from rbee-ui to commercial..."
echo ""

# Step 1: Create pages directory in commercial
echo "📁 Step 1: Creating pages directory in commercial site..."
mkdir -p /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages
echo "✅ Directory created"
echo ""

# Step 2: Copy all pages
echo "📦 Step 2: Copying all 18 page directories..."
cp -r /home/vince/Projects/llama-orch/frontend/packages/rbee-ui/src/pages/* \
      /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages/
echo "✅ Pages copied"
echo ""

# Step 3: Verify the copy
echo "🔍 Step 3: Verifying migration..."
PAGES_COUNT=$(ls -1 /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages/ | wc -l)
echo "   Found $PAGES_COUNT items (should be 19: 18 directories + 1 index.ts)"
echo ""
echo "   Pages migrated:"
ls -1 /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages/
echo ""

# Step 4: Initialize git repo in commercial
echo "🔧 Step 4: Initializing git repo in commercial site..."
cd /home/vince/Projects/llama-orch/frontend/apps/commercial

if [ -d .git ]; then
    echo "⚠️  Git repo already exists, skipping initialization"
else
    git init
    echo "✅ Git initialized"
fi
echo ""

# Step 5: Create initial commit
echo "💾 Step 5: Creating initial commit..."
git add .
git commit -m "Initial commit: Commercial site with migrated pages

- Migrated 18 page directories from rbee-ui
- PageProps files now private (CMS content)
- Commercial Next.js app structure
- Total: ~60 page component files

Pages migrated:
- CommunityPage, CompliancePage, DevOpsPage, DevelopersPage
- EducationPage, EnterprisePage, FeaturesPage, HomePage
- HomelabPage, LegalPage, PricingPage, PrivacyPage
- ProvidersPage, ResearchPage, SecurityPage, StartupsPage
- TermsPage, UseCasesPage

This keeps marketing content private while UI library stays open-source."
echo "✅ Initial commit created"
echo ""

# Step 6: Add remote and push
echo "🌐 Step 6: Adding remote and pushing to private repo..."
if git remote | grep -q origin; then
    echo "⚠️  Remote 'origin' already exists"
    echo "   Current remote: $(git remote get-url origin)"
else
    git remote add origin git@github.com:veighnsche/rbee-commercial-private.git
    echo "✅ Remote added"
fi

git branch -M main
echo ""
echo "📤 Ready to push! Run this command manually:"
echo "   cd /home/vince/Projects/llama-orch/frontend/apps/commercial"
echo "   git push -u origin main"
echo ""

# Step 7: Instructions for cleanup
echo "📋 Next steps:"
echo ""
echo "1. Push to private repo:"
echo "   cd /home/vince/Projects/llama-orch/frontend/apps/commercial"
echo "   git push -u origin main"
echo ""
echo "2. Remove pages from rbee-ui (public package):"
echo "   cd /home/vince/Projects/llama-orch"
echo "   rm -rf frontend/packages/rbee-ui/src/pages"
echo "   # Edit frontend/packages/rbee-ui/src/index.ts and remove: export * from './pages'"
echo ""
echo "3. Set up submodule in main repo:"
echo "   cd /home/vince/Projects/llama-orch"
echo "   git rm -r --cached frontend/apps/commercial"
echo "   git submodule add git@github.com:veighnsche/rbee-commercial-private.git frontend/apps/commercial"
echo "   git add .gitignore .gitmodules frontend/apps/commercial"
echo "   git commit -m 'Move commercial site to private submodule'"
echo ""
echo "✅ Migration preparation complete!"
echo ""
echo "⚠️  IMPORTANT: Review COMMERCIAL_SUBMODULE_SETUP.md for full details"
