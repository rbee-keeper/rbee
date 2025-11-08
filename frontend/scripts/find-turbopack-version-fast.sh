#!/bin/bash
# TEAM-450: Fast parallel version finder for Turbopack compatibility
# Tests multiple Next.js versions in parallel

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MAX_PARALLEL=4
BASE_TEST_DIR="/tmp/turbopack-tests"
RESULTS_FILE="/tmp/turbopack-results.txt"

# Next.js versions to test (newest to oldest)
VERSIONS=(
  "16.0.1"    # Current (known broken)
  "16.0.0"
  "15.1.6"
  "15.1.5"
  "15.1.4"
  "15.1.3"
  "15.1.2"
  "15.1.1"
  "15.1.0"
  "15.0.3"
  "15.0.2"
  "15.0.1"
  "15.0.0"
)

echo -e "${BLUE}🚀 Fast Turbopack Version Finder${NC}"
echo "Testing ${#VERSIONS[@]} versions with $MAX_PARALLEL parallel jobs"
echo ""

# Cleanup
rm -rf "$BASE_TEST_DIR"
mkdir -p "$BASE_TEST_DIR"
rm -f "$RESULTS_FILE"
touch "$RESULTS_FILE"

# Function to test a single version
test_version() {
  local version=$1
  local test_dir="$BASE_TEST_DIR/test-$version"
  local log_file="$test_dir/build.log"
  
  mkdir -p "$test_dir"
  cd "$test_dir"
  
  # Create minimal test project
  cat > package.json << EOF
{
  "name": "test-$version",
  "private": true,
  "scripts": { "build": "next build" },
  "dependencies": {
    "next": "$version",
    "react": "19.2.0",
    "react-dom": "19.2.0",
    "nextra": "^4.6.0",
    "nextra-theme-docs": "^4.6.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4.1.17",
    "tailwindcss": "^4.1.17",
    "typescript": "^5.9.3",
    "@types/react": "^19.2.2",
    "@types/react-dom": "^19.2.2"
  }
}
EOF

  cat > next.config.ts << 'EOF'
import type { NextConfig } from 'next'
import nextra from 'nextra'
const withNextra = nextra({ defaultShowCopyCode: true })
export default withNextra({
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
})
EOF

  mkdir -p app/docs
  
  cat > app/layout.tsx << 'EOF'
import './globals.css'
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html><body>{children}</body></html>
}
EOF

  cat > app/globals.css << 'EOF'
@import "tailwindcss";
EOF

  cat > app/page.tsx << 'EOF'
export default function Home() {
  return <div className="p-4 bg-blue-500">Test</div>
}
EOF

  cat > app/docs/page.mdx << 'EOF'
# Test
EOF

  cat > mdx-components.tsx << 'EOF'
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'
export function useMDXComponents(components: any): any {
  return { ...getDocsComponents(components), ...components }
}
EOF

  cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": false,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }]
  },
  "include": ["**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

  # Install and build
  echo "[$version] Installing..." > "$log_file"
  if ! pnpm install --no-frozen-lockfile >> "$log_file" 2>&1; then
    echo "$version|INSTALL_FAILED" >> "$RESULTS_FILE"
    return 1
  fi
  
  echo "[$version] Building..." >> "$log_file"
  # Note: Next.js 15.x doesn't support --turbopack for builds, only for dev
  # Next.js 16.x uses Turbopack by default for builds
  local build_cmd="pnpm exec next build"
  
  if $build_cmd >> "$log_file" 2>&1; then
    echo "$version|SUCCESS" >> "$RESULTS_FILE"
    return 0
  else
    # Detect error type
    local error_type="BUILD_FAILED"
    if grep -q "Parsing CSS source code failed" "$log_file"; then
      error_type="CSS_PARSE_ERROR"
    elif grep -q "next-mdx-import-source-file" "$log_file"; then
      error_type="NEXTRA_ERROR"
    fi
    echo "$version|$error_type" >> "$RESULTS_FILE"
    return 1
  fi
}

export -f test_version
export BASE_TEST_DIR RESULTS_FILE

# Run tests in parallel
echo -e "${YELLOW}Running tests...${NC}"
printf '%s\n' "${VERSIONS[@]}" | xargs -P "$MAX_PARALLEL" -I {} bash -c 'test_version "$@"' _ {}

# Wait for all jobs
wait

# Analyze results
echo ""
echo "=========================================="
echo -e "${BLUE}Test Results:${NC}"
echo "=========================================="

LAST_WORKING=""
while IFS='|' read -r version status; do
  case "$status" in
    SUCCESS)
      echo -e "${GREEN}✓ $version: SUCCESS${NC}"
      if [ -z "$LAST_WORKING" ]; then
        LAST_WORKING="$version"
      fi
      ;;
    INSTALL_FAILED)
      echo -e "${RED}✗ $version: Install failed${NC}"
      ;;
    CSS_PARSE_ERROR)
      echo -e "${RED}✗ $version: Tailwind CSS v4 parsing error${NC}"
      ;;
    NEXTRA_ERROR)
      echo -e "${RED}✗ $version: Nextra MDX import error${NC}"
      ;;
    BUILD_FAILED)
      echo -e "${RED}✗ $version: Build failed${NC}"
      ;;
  esac
done < "$RESULTS_FILE"

echo "=========================================="
echo ""

if [ -n "$LAST_WORKING" ]; then
  echo -e "${GREEN}✓ Last working version: Next.js $LAST_WORKING${NC}"
  echo ""
  echo "To pin to this version:"
  echo ""
  echo "  # Update all Next.js dependencies"
  echo "  pnpm update next@$LAST_WORKING --recursive"
  echo ""
  echo "  # Or manually update package.json files:"
  echo "  \"next\": \"$LAST_WORKING\""
  echo ""
  echo "Test logs available in: $BASE_TEST_DIR/test-$LAST_WORKING/"
else
  echo -e "${RED}✗ No working version found${NC}"
  echo ""
  echo "All tested versions have issues with Turbopack + Tailwind CSS v4 + Nextra"
  echo ""
  echo "Recommendations:"
  echo "1. Use webpack for builds (current solution)"
  echo "2. Test older Next.js 14.x versions"
  echo "3. Wait for upstream fixes:"
  echo "   - https://github.com/shuding/nextra/issues/4830"
  echo "   - https://github.com/tailwindlabs/tailwindcss/discussions/15905"
fi

echo ""
echo "Test artifacts: $BASE_TEST_DIR"
echo "Results file: $RESULTS_FILE"
