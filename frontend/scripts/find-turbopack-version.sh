#!/bin/bash
# TEAM-450: Find last working Next.js version with Turbopack
# Tests Next.js versions to find where Turbopack + Tailwind CSS v4 + Nextra work together

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="/tmp/turbopack-version-test"
LOG_FILE="/tmp/turbopack-test-results.log"

# Next.js versions to test (newest to oldest)
VERSIONS=(
  "16.0.1"    # Current (broken)
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
  "15.0.0-rc.1"
  "15.0.0-canary.179"
)

echo "🔍 Finding last working Next.js version with Turbopack"
echo "Testing versions: ${VERSIONS[*]}"
echo ""
echo "Results will be logged to: $LOG_FILE"
echo "" > "$LOG_FILE"

# Function to test a version
test_version() {
  local version=$1
  echo -e "${YELLOW}Testing Next.js $version...${NC}"
  echo "=== Testing Next.js $version ===" >> "$LOG_FILE"
  
  # Clean test directory
  rm -rf "$TEST_DIR"
  mkdir -p "$TEST_DIR"
  cd "$TEST_DIR"
  
  # Create minimal test project
  cat > package.json << EOF
{
  "name": "turbopack-test",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "build": "next build"
  },
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

  # Create next.config.ts with Nextra
  cat > next.config.ts << 'EOF'
import type { NextConfig } from 'next'
import nextra from 'nextra'

const nextConfig: NextConfig = {
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
}

const withNextra = nextra({
  defaultShowCopyCode: true,
})

export default withNextra(nextConfig)
EOF

  # Create app directory structure
  mkdir -p app/docs
  
  # Create layout.tsx
  cat > app/layout.tsx << 'EOF'
import './globals.css'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
EOF

  # Create globals.css with Tailwind v4
  cat > app/globals.css << 'EOF'
@import "tailwindcss";
EOF

  # Create a simple page
  cat > app/page.tsx << 'EOF'
export default function Home() {
  return <div className="p-4 bg-blue-500">Test</div>
}
EOF

  # Create docs page
  cat > app/docs/page.mdx << 'EOF'
# Test Documentation

This is a test page with Nextra.
EOF

  # Create mdx-components.tsx
  cat > mdx-components.tsx << 'EOF'
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    ...components,
  }
}
EOF

  # Create tsconfig.json
  cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
EOF

  # Install dependencies
  echo "  📦 Installing dependencies..." >> "$LOG_FILE"
  if ! pnpm install --no-frozen-lockfile >> "$LOG_FILE" 2>&1; then
    echo -e "  ${RED}✗ Install failed${NC}"
    echo "  Result: INSTALL_FAILED" >> "$LOG_FILE"
    return 1
  fi
  
  # Test build with Turbopack (default in Next.js 16, flag in 15)
  echo "  🔨 Building with Turbopack..." >> "$LOG_FILE"
  
  # Determine if we need --turbopack flag
  local build_cmd="next build"
  if [[ "$version" == 15.* ]] || [[ "$version" == *"canary"* ]] || [[ "$version" == *"rc"* ]]; then
    build_cmd="next build --turbopack"
  fi
  
  if $build_cmd >> "$LOG_FILE" 2>&1; then
    echo -e "  ${GREEN}✓ Build successful${NC}"
    echo "  Result: SUCCESS" >> "$LOG_FILE"
    echo ""
    return 0
  else
    echo -e "  ${RED}✗ Build failed${NC}"
    echo "  Result: BUILD_FAILED" >> "$LOG_FILE"
    
    # Check for specific errors
    if grep -q "Parsing CSS source code failed" "$LOG_FILE"; then
      echo "  Error: Tailwind CSS v4 parsing error" >> "$LOG_FILE"
    fi
    if grep -q "next-mdx-import-source-file" "$LOG_FILE"; then
      echo "  Error: Nextra MDX import error" >> "$LOG_FILE"
    fi
    
    echo ""
    return 1
  fi
}

# Test each version
LAST_WORKING=""
for version in "${VERSIONS[@]}"; do
  if test_version "$version"; then
    LAST_WORKING="$version"
    echo -e "${GREEN}✓ Next.js $version works with Turbopack!${NC}"
    break
  fi
done

# Cleanup
rm -rf "$TEST_DIR"

# Report results
echo ""
echo "=========================================="
if [ -n "$LAST_WORKING" ]; then
  echo -e "${GREEN}✓ Last working version: Next.js $LAST_WORKING${NC}"
  echo ""
  echo "To pin to this version, update package.json:"
  echo "  \"next\": \"$LAST_WORKING\""
  echo ""
  echo "Full test log: $LOG_FILE"
else
  echo -e "${RED}✗ No working version found in tested range${NC}"
  echo ""
  echo "Options:"
  echo "1. Test older versions (add to VERSIONS array)"
  echo "2. Use webpack for builds (current solution)"
  echo "3. Wait for upstream fixes"
  echo ""
  echo "Full test log: $LOG_FILE"
fi
echo "=========================================="
