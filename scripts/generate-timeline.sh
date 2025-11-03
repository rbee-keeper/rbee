#!/usr/bin/env bash
# Quick wrapper to generate repository timeline for investors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "🚀 Generating Repository Timeline for Investors"
echo ""

# Run the Python script
python3 "$SCRIPT_DIR/generate-hourly-timeline.py" "$@"

echo ""
echo "📁 Output location: .timeline/"
echo "📖 Start here: .timeline/INDEX.md"
