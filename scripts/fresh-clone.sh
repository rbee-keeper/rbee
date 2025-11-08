#!/bin/bash
# Fresh Clone Script - Completely delete and reclone the repository
# SAFETY: Only runs if everything is committed (no uncommitted changes)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  FRESH CLONE SCRIPT${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Get the current directory (should be llama-orch root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
PARENT_DIR="$(dirname "$REPO_DIR")"

echo -e "Repository: ${GREEN}$REPO_DIR${NC}"
echo -e "Parent dir: ${GREEN}$PARENT_DIR${NC}"
echo ""

# Change to repo directory
cd "$REPO_DIR"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAFETY CHECK 1: Verify we're in a git repository
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ! -d ".git" ]; then
    echo -e "${RED}ERROR: Not in a git repository!${NC}"
    echo "This script must be run from the repository root."
    exit 1
fi

echo -e "${GREEN}✓${NC} Git repository detected"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAFETY CHECK 2: Verify everything is committed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "Checking for uncommitted changes..."

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ERROR: UNCOMMITTED CHANGES DETECTED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "You have uncommitted changes. Please commit or stash them first:"
    echo ""
    git status --short
    echo ""
    echo -e "${YELLOW}To commit:${NC}"
    echo "  git add ."
    echo "  git commit -m \"Your commit message\""
    echo ""
    echo -e "${YELLOW}To stash:${NC}"
    echo "  git stash"
    echo ""
    exit 1
fi

# Check for untracked files
UNTRACKED=$(git ls-files --others --exclude-standard)
if [ -n "$UNTRACKED" ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ERROR: UNTRACKED FILES DETECTED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "You have untracked files. Please add and commit them first:"
    echo ""
    echo "$UNTRACKED"
    echo ""
    echo -e "${YELLOW}To add and commit:${NC}"
    echo "  git add ."
    echo "  git commit -m \"Add untracked files\""
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} No uncommitted changes"
echo -e "${GREEN}✓${NC} No untracked files"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAFETY CHECK 3: Get remote URL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REMOTE_URL=$(git config --get remote.origin.url)
if [ -z "$REMOTE_URL" ]; then
    echo -e "${RED}ERROR: No remote origin URL found!${NC}"
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    echo -e "${RED}ERROR: Could not determine current branch!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Remote URL: $REMOTE_URL"
echo -e "${GREEN}✓${NC} Current branch: $CURRENT_BRANCH"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL CONFIRMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  WARNING: THIS WILL DELETE EVERYTHING${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "This script will:"
echo "  1. Delete the entire directory: $REPO_DIR"
echo "  2. Clone fresh from: $REMOTE_URL"
echo "  3. Checkout branch: $CURRENT_BRANCH"
echo ""
echo -e "${RED}ALL local files will be PERMANENTLY DELETED!${NC}"
echo ""
read -p "Are you ABSOLUTELY SURE? Type 'yes' to continue: " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo ""
    echo -e "${YELLOW}Aborted. No changes made.${NC}"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DELETE AND RECLONE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo -e "${YELLOW}Starting fresh clone...${NC}"
echo ""

# Change to parent directory
cd "$PARENT_DIR"

# Delete the repository
echo "Deleting $REPO_DIR..."
rm -rf "$REPO_DIR"
echo -e "${GREEN}✓${NC} Deleted"

# Clone fresh with submodules
echo ""
echo "Cloning from $REMOTE_URL..."
git clone --recurse-submodules "$REMOTE_URL" "$REPO_NAME"
echo -e "${GREEN}✓${NC} Cloned"

# Change to new repo
cd "$REPO_NAME"

# Checkout the branch if not main/master
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    echo ""
    echo "Checking out branch: $CURRENT_BRANCH..."
    git checkout "$CURRENT_BRANCH"
    echo -e "${GREEN}✓${NC} Checked out $CURRENT_BRANCH"
fi

# Initialize and update submodules (in case they weren't cloned with --recurse-submodules)
echo ""
echo "Initializing and updating submodules..."
git submodule update --init --recursive
echo -e "${GREEN}✓${NC} Submodules updated"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUCCESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  FRESH CLONE COMPLETE!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Repository: $REPO_DIR"
echo "Branch: $CURRENT_BRANCH"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  cd $REPO_DIR"
echo "  cargo build --release"
echo "  # or"
echo "  cd frontend && pnpm install && pnpm build"
echo ""
