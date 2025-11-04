#!/bin/bash
# TEAM-405: Cleanup script - Delete unused marketplace search files

set -e

echo "🧹 TEAM-405: Cleaning up unused marketplace search files..."

# ModelManagement - Remove marketplace search files
echo "📁 Removing ModelManagement marketplace search files..."
rm -f bin/20_rbee_hive/ui/app/src/components/ModelManagement/SearchResultsView.tsx
rm -f bin/20_rbee_hive/ui/app/src/components/ModelManagement/FilterPanel.tsx
rm -f bin/20_rbee_hive/ui/app/src/components/ModelManagement/utils.ts

# WorkerManagement - Remove marketplace catalog files
echo "📁 Removing WorkerManagement marketplace catalog files..."
rm -f bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx

echo "✅ Cleanup complete!"
echo ""
echo "📊 Files removed:"
echo "  - ModelManagement/SearchResultsView.tsx (207 lines)"
echo "  - ModelManagement/FilterPanel.tsx (160 lines)"
echo "  - ModelManagement/utils.ts (~100 lines)"
echo "  - WorkerManagement/WorkerCatalogView.tsx (410 lines)"
echo ""
echo "Total: ~877 lines removed"
echo ""
echo "🔍 Verify with: cd bin/20_rbee_hive/ui/app && pnpm tsc -b --noEmit"
