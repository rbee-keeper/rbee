# Checklist 01: Shared Components Package

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** None (can start immediately)

---

## 🎯 Goal

Create `@rbee/marketplace-components` package with dumb, reusable components that work in both Next.js (SSG) and Tauri.

---

## 📦 Phase 1: Package Setup (Day 1)

### 1.1 Create Package Structure

- [ ] Create directory: `frontend/packages/marketplace-components/`
- [ ] Create `package.json`:
  ```json
  {
    "name": "@rbee/marketplace-components",
    "version": "1.0.0",
    "main": "./dist/index.js",
    "types": "./dist/index.d.ts",
    "exports": {
      ".": "./dist/index.js",
      "./components/*": "./dist/components/*.js"
    },
    "scripts": {
      "build": "tsup src/index.ts --format esm,cjs --dts",
      "dev": "tsup src/index.ts --format esm,cjs --dts --watch"
    },
    "peerDependencies": {
      "react": "^18.0.0",
      "react-dom": "^18.0.0"
    },
    "devDependencies": {
      "@types/react": "^18.0.0",
      "tsup": "^8.0.0",
      "typescript": "^5.0.0"
    }
  }
  ```
- [ ] Create `tsconfig.json`:
  ```json
  {
    "compilerOptions": {
      "target": "ES2020",
      "module": "ESNext",
      "lib": ["ES2020", "DOM"],
      "jsx": "react-jsx",
      "declaration": true,
      "outDir": "./dist",
      "strict": true,
      "moduleResolution": "node",
      "esModuleInterop": true,
      "skipLibCheck": true
    },
    "include": ["src"],
    "exclude": ["node_modules", "dist"]
  }
  ```
- [ ] Create `src/` directory
- [ ] Create `src/index.ts` (empty for now)
- [ ] Run `pnpm install` in package directory
- [ ] Verify build works: `pnpm build`

### 1.2 Create Type Definitions

- [ ] Create `src/types/model.ts`:
  ```typescript
  export interface Model {
    id: string
    name: string
    description: string
    author?: string
    imageUrl?: string
    tags: string[]
    downloads: number
    likes: number
    size: string
    source: 'huggingface' | 'civitai'
  }
  ```
- [ ] Create `src/types/worker.ts`:
  ```typescript
  export interface Worker {
    id: string
    name: string
    description: string
    version: string
    platform: string[]
    architecture: string[]
    workerType: 'cpu' | 'cuda' | 'metal'
  }
  ```
- [ ] Create `src/types/marketplace.ts`:
  ```typescript
  export type MarketplaceMode = 'nextjs' | 'tauri'
  
  export interface MarketplaceFilters {
    search: string
    category?: string
    sort: 'popular' | 'recent' | 'trending'
  }
  ```
- [ ] Export all types from `src/types/index.ts`
- [ ] Add to `src/index.ts`: `export * from './types'`

---

## 🎨 Phase 2: Core Components (Days 2-3)

### 2.1 ModelCard Component

- [ ] Create `src/components/ModelCard.tsx`
- [ ] Define props interface:
  ```typescript
  interface ModelCardProps {
    model: Model
    onDownload?: (modelId: string) => void
    downloadButton?: React.ReactNode
    mode?: MarketplaceMode
  }
  ```
- [ ] Implement component structure:
  - [ ] Card container with hover effect
  - [ ] Image/thumbnail display (with fallback)
  - [ ] Model name (h3)
  - [ ] Author name (if present)
  - [ ] Description (truncated to 2 lines)
  - [ ] Tags (first 3, with badge styling)
  - [ ] Stats row (downloads, likes, size)
  - [ ] Download button slot (customizable)
- [ ] Add CSS classes (Tailwind):
  - [ ] `model-card` - Main container
  - [ ] `model-card-image` - Image container
  - [ ] `model-card-content` - Content area
  - [ ] `model-card-stats` - Stats row
  - [ ] `model-card-actions` - Button area
- [ ] Test with mock data
- [ ] Verify works in both modes (nextjs/tauri)

### 2.2 WorkerCard Component

- [ ] Create `src/components/WorkerCard.tsx`
- [ ] Define props interface:
  ```typescript
  interface WorkerCardProps {
    worker: Worker
    onInstall?: (workerId: string) => void
    installButton?: React.ReactNode
    mode?: MarketplaceMode
  }
  ```
- [ ] Implement component structure:
  - [ ] Card container
  - [ ] Worker icon (based on type: CPU/CUDA/Metal)
  - [ ] Worker name
  - [ ] Description
  - [ ] Version badge
  - [ ] Platform badges (Linux/macOS/Windows)
  - [ ] Architecture badges (x86_64/aarch64)
  - [ ] Install button slot
- [ ] Add CSS classes
- [ ] Test with mock data
- [ ] Verify works in both modes

### 2.3 MarketplaceGrid Component

- [ ] Create `src/components/MarketplaceGrid.tsx`
- [ ] Define props interface:
  ```typescript
  interface MarketplaceGridProps<T> {
    items: T[]
    renderItem: (item: T) => React.ReactNode
    isLoading?: boolean
    error?: string
    emptyMessage?: string
  }
  ```
- [ ] Implement component:
  - [ ] Loading state (skeleton cards)
  - [ ] Error state (error message with retry)
  - [ ] Empty state (empty message with icon)
  - [ ] Grid layout (responsive: 1/2/3/4 columns)
  - [ ] Item rendering (using renderItem prop)
- [ ] Add CSS classes:
  - [ ] `marketplace-grid` - Grid container
  - [ ] `marketplace-grid-loading` - Loading state
  - [ ] `marketplace-grid-error` - Error state
  - [ ] `marketplace-grid-empty` - Empty state
- [ ] Test with different item counts
- [ ] Test loading/error/empty states

### 2.4 FilterSidebar Component

- [ ] Create `src/components/FilterSidebar.tsx`
- [ ] Define props interface:
  ```typescript
  interface FilterSidebarProps {
    filters: MarketplaceFilters
    onFilterChange: (filters: MarketplaceFilters) => void
    categories?: Array<{ value: string; label: string }>
    sortOptions?: Array<{ value: string; label: string }>
  }
  ```
- [ ] Implement component:
  - [ ] Search input (with icon)
  - [ ] Category dropdown (if provided)
  - [ ] Sort dropdown
  - [ ] Clear filters button
  - [ ] Active filters chips
- [ ] Add CSS classes
- [ ] Test filter changes
- [ ] Test clear functionality

### 2.5 SearchBar Component

- [ ] Create `src/components/SearchBar.tsx`
- [ ] Define props interface:
  ```typescript
  interface SearchBarProps {
    value: string
    onChange: (value: string) => void
    placeholder?: string
  }
  ```
- [ ] Implement component:
  - [ ] Input field
  - [ ] Search icon (left)
  - [ ] Clear button (right, when value exists)
  - [ ] Debounced onChange (300ms)
- [ ] Add CSS classes
- [ ] Test typing and clearing
- [ ] Test debounce behavior

---

## 🔧 Phase 3: Utility Components (Day 4)

### 3.1 LoadingSpinner Component

- [ ] Create `src/components/LoadingSpinner.tsx`
- [ ] Implement spinner animation (CSS)
- [ ] Add size variants (sm, md, lg)
- [ ] Test in different contexts

### 3.2 ErrorMessage Component

- [ ] Create `src/components/ErrorMessage.tsx`
- [ ] Define props:
  ```typescript
  interface ErrorMessageProps {
    message: string
    onRetry?: () => void
  }
  ```
- [ ] Implement component:
  - [ ] Error icon
  - [ ] Error message
  - [ ] Retry button (if onRetry provided)
- [ ] Test with different messages

### 3.3 EmptyState Component

- [ ] Create `src/components/EmptyState.tsx`
- [ ] Define props:
  ```typescript
  interface EmptyStateProps {
    message: string
    icon?: React.ReactNode
    action?: React.ReactNode
  }
  ```
- [ ] Implement component:
  - [ ] Icon (default or custom)
  - [ ] Message
  - [ ] Optional action button
- [ ] Test with different configurations

### 3.4 Badge Component

- [ ] Create `src/components/Badge.tsx`
- [ ] Define props:
  ```typescript
  interface BadgeProps {
    children: React.ReactNode
    variant?: 'default' | 'primary' | 'success' | 'warning'
  }
  ```
- [ ] Implement component with variants
- [ ] Test all variants

---

## 📤 Phase 4: Export & Documentation (Day 5)

### 4.1 Update Exports

- [ ] Update `src/index.ts`:
  ```typescript
  // Types
  export * from './types'
  
  // Components
  export { ModelCard } from './components/ModelCard'
  export { WorkerCard } from './components/WorkerCard'
  export { MarketplaceGrid } from './components/MarketplaceGrid'
  export { FilterSidebar } from './components/FilterSidebar'
  export { SearchBar } from './components/SearchBar'
  export { LoadingSpinner } from './components/LoadingSpinner'
  export { ErrorMessage } from './components/ErrorMessage'
  export { EmptyState } from './components/EmptyState'
  export { Badge } from './components/Badge'
  ```
- [ ] Build package: `pnpm build`
- [ ] Verify all exports work

### 4.2 Create README

- [ ] Create `README.md`:
  ```markdown
  # @rbee/marketplace-components
  
  Shared marketplace components for Next.js and Tauri.
  
  ## Installation
  
  \`\`\`bash
  pnpm add @rbee/marketplace-components
  \`\`\`
  
  ## Usage
  
  ### ModelCard
  
  \`\`\`tsx
  import { ModelCard } from '@rbee/marketplace-components'
  
  <ModelCard
    model={model}
    onDownload={(id) => console.log('Download', id)}
    mode="nextjs"
  />
  \`\`\`
  
  ### WorkerCard
  
  \`\`\`tsx
  import { WorkerCard } from '@rbee/marketplace-components'
  
  <WorkerCard
    worker={worker}
    onInstall={(id) => console.log('Install', id)}
    mode="tauri"
  />
  \`\`\`
  
  ## Components
  
  - ModelCard - Display model information
  - WorkerCard - Display worker information
  - MarketplaceGrid - Grid layout with loading/error states
  - FilterSidebar - Filter and search controls
  - SearchBar - Search input with debounce
  - LoadingSpinner - Loading indicator
  - ErrorMessage - Error display with retry
  - EmptyState - Empty state display
  - Badge - Tag/badge component
  
  ## Modes
  
  Components support two modes:
  - `nextjs` - For Next.js SSG/SSR
  - `tauri` - For Tauri desktop app
  
  The only difference is how buttons behave (links vs callbacks).
  ```
- [ ] Add examples for each component

### 4.3 Create Storybook (Optional)

- [ ] Install Storybook: `pnpm add -D @storybook/react`
- [ ] Create `.storybook/main.js`
- [ ] Create stories for each component:
  - [ ] `ModelCard.stories.tsx`
  - [ ] `WorkerCard.stories.tsx`
  - [ ] `MarketplaceGrid.stories.tsx`
  - [ ] `FilterSidebar.stories.tsx`
  - [ ] `SearchBar.stories.tsx`
- [ ] Run Storybook: `pnpm storybook`
- [ ] Verify all components render correctly

---

## ✅ Phase 5: Testing (Days 6-7)

### 5.1 Unit Tests

- [ ] Install testing dependencies:
  ```bash
  pnpm add -D @testing-library/react @testing-library/jest-dom vitest
  ```
- [ ] Create `vitest.config.ts`
- [ ] Write tests for ModelCard:
  - [ ] Renders model data correctly
  - [ ] Calls onDownload when button clicked
  - [ ] Shows custom download button if provided
  - [ ] Handles missing image gracefully
- [ ] Write tests for WorkerCard:
  - [ ] Renders worker data correctly
  - [ ] Calls onInstall when button clicked
  - [ ] Shows correct platform badges
- [ ] Write tests for MarketplaceGrid:
  - [ ] Shows loading state
  - [ ] Shows error state
  - [ ] Shows empty state
  - [ ] Renders items correctly
- [ ] Write tests for FilterSidebar:
  - [ ] Updates filters on change
  - [ ] Clears filters correctly
- [ ] Write tests for SearchBar:
  - [ ] Debounces input
  - [ ] Shows clear button when has value
  - [ ] Clears value when clear clicked
- [ ] Run tests: `pnpm test`
- [ ] Verify all tests pass

### 5.2 Integration Tests

- [ ] Create test Next.js app in `examples/nextjs/`
- [ ] Install package: `pnpm add @rbee/marketplace-components`
- [ ] Create test page using ModelCard
- [ ] Create test page using WorkerCard
- [ ] Create test page using MarketplaceGrid
- [ ] Verify SSG works: `pnpm build`
- [ ] Verify components render correctly
- [ ] Create test Tauri app in `examples/tauri/`
- [ ] Install package
- [ ] Create test page using components
- [ ] Verify Tauri build works
- [ ] Verify components render correctly

### 5.3 Visual Regression Tests (Optional)

- [ ] Install Playwright: `pnpm add -D @playwright/test`
- [ ] Create visual tests for each component
- [ ] Take baseline screenshots
- [ ] Run tests: `pnpm test:visual`
- [ ] Verify no regressions

---

## 📊 Success Criteria

### Must Have

- [ ] All 9 components implemented
- [ ] All components work in Next.js
- [ ] All components work in Tauri
- [ ] Package builds successfully
- [ ] All exports work correctly
- [ ] README with examples
- [ ] Unit tests pass

### Nice to Have

- [ ] Storybook with all components
- [ ] Integration tests with Next.js
- [ ] Integration tests with Tauri
- [ ] Visual regression tests
- [ ] 100% test coverage

---

## 🚀 Deliverables

1. **Package:** `@rbee/marketplace-components` published to workspace
2. **Components:** 9 reusable components
3. **Types:** Complete TypeScript definitions
4. **Tests:** Unit tests for all components
5. **Documentation:** README with examples
6. **Examples:** Test apps in Next.js and Tauri

---

## 📝 Notes

### Key Principles

1. **DUMB COMPONENTS** - No data fetching, only props
2. **MODE AGNOSTIC** - Work in both Next.js and Tauri
3. **CUSTOMIZABLE** - Accept custom buttons/actions
4. **TYPED** - Full TypeScript support
5. **TESTED** - Unit tests for all components

### Common Pitfalls

- ❌ Don't fetch data in components
- ❌ Don't use Next.js-specific features (Link, Image)
- ❌ Don't use Tauri-specific features (invoke)
- ❌ Don't hardcode styles (use Tailwind classes)
- ✅ Accept data via props
- ✅ Accept callbacks via props
- ✅ Accept custom buttons via props
- ✅ Use generic React components

---

**Start with Phase 1, complete each checkbox in order!** ✅
