#!/bin/bash
# Configure branch protection for rbee repository
# Created by: TEAM-451
#
# Requirements:
# - production: Strict protection, triggers releases
# - development: Free pushing, no restrictions

set -e

REPO="rbee-keeper/rbee"
PROD_BRANCH="production"
DEV_BRANCH="development"

echo "🛡️  Configuring branch protection for $REPO..."
echo ""

# Production Branch - Strict Protection
echo "📦 Configuring production branch (release trigger)..."

gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/$REPO/branches/$PROD_BRANCH/protection" \
    --input - << 'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["build", "test", "clippy"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
EOF

if [ $? -eq 0 ]; then
    echo "✅ Production branch protected"
else
    echo "⚠️  Production protection failed (may need admin access)"
fi

echo ""

# Development Branch - No Protection (Free Pushing)
echo "🚀 Configuring development branch (free pushing)..."

# Remove any existing protection
gh api \
    --method DELETE \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/$REPO/branches/$DEV_BRANCH/protection" \
    2>/dev/null || echo "No existing protection to remove"

echo "✅ Development branch - free pushing enabled"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Branch protection configured!"
echo ""
echo "📋 Configuration:"
echo "   • production  - Protected (PR required, CI must pass, triggers releases)"
echo "   • development - Free pushing (no restrictions)"
echo ""
echo "🔄 Workflow:"
echo "   1. Push freely to development"
echo "   2. Create PR from development → production when ready to release"
echo "   3. Merge triggers production deployment"
echo ""
echo "🔍 Verify protection:"
echo "   gh api repos/$REPO/branches/$PROD_BRANCH/protection | jq"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
