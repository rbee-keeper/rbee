# Checklist 01: Marketplace Components (rbee-ui)

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** None (can start immediately)  
**TEAM-400:** ✅ Updated to use existing rbee-ui package

---

## 🎯 Goal

Create marketplace components in `frontend/packages/rbee-ui/src/marketplace/` using atomic design pattern. Components are DUMB, work in both Next.js (SSG) and Tauri, and follow existing architecture.

---

## 📦 Phase 1: Setup Marketplace Directory (Day 1 Morning)

### 1.1 Create Directory Structure

**TEAM-400:** We use `rbee-ui/src/marketplace/` NOT a separate package.

- [ ] Navigate to: `frontend/packages/rbee-ui/src/marketplace/`
- [ ] Create subdirectories:
  ```bash
  mkdir -p organisms templates pages
  ```
- [ ] Verify structure:
  ```
  rbee-ui/src/marketplace/
  ├── organisms/   (NEW - ModelCard, WorkerCard, etc.)
  ├── templates/   (NEW - ModelListTemplate, ModelDetailTemplate)
  └── pages/       (NEW - ModelsPage, ModelDetailPage)
  ```

### 1.2 Understand Existing rbee-ui Structure

**TEAM-400:** Study the existing pattern before building.

- [ ] Review atomic design structure:
  ```
  rbee-ui/src/
  ├── atoms/        (Button, Badge, Card - REUSE THESE)
  ├── molecules/    (StatsGrid, TerminalWindow - REUSE THESE)
  ├── organisms/    (Navbars, complex components)
  ├── templates/    (Page sections - HeroTemplate, FAQTemplate)
  └── marketplace/  (Our new components HERE)
  ```
- [ ] Study commercial site pattern: `frontend/apps/commercial/components/pages/HomePage/`
  - [ ] Read `HomePage.tsx` - DUMB page (just renders templates)
  - [ ] Read `HomePageProps.tsx` - ALL data in Props file (perfect for SSG)
  - [ ] Understand template wrapping with `TemplateContainer`
- [ ] Note: We REUSE atoms/molecules, CREATE marketplace-specific organisms/templates/pages

### 1.3 Review Consistency Requirements

**TEAM-400:** User emphasizes consistency (see memory).

- [ ] Read existing Card components in `rbee-ui/src/atoms/Card/`
- [ ] Note consistent card structure:
  - Card with padding (p-6 or p-8)
  - IconCardHeader (if has header)
  - CardContent with p-0
  - Optional CardFooter
- [ ] DON'T mix patterns - use same structure everywhere

---

## 🎨 Phase 2: Organisms - Cards (Days 1-2)

**TEAM-400:** Organisms are marketplace-specific reusable components.

### 2.1 ModelCard Organism

- [ ] Create directory: `rbee-ui/src/marketplace/organisms/ModelCard/`
- [ ] Create `ModelCard.tsx`:
  ```tsx
  // TEAM-400: Marketplace model card organism
  import { Card } from '@rbee/ui/atoms/Card'
  import { Badge } from '@rbee/ui/atoms/Badge'
  import { Button } from '@rbee/ui/atoms/Button'
  
  export interface ModelCardProps {
    model: {
      id: string
      name: string
      description: string
      author?: string
      imageUrl?: string
      tags: string[]
      downloads: number
      likes: number
      size: string
    }
    onAction?: (modelId: string) => void
    actionButton?: React.ReactNode
  }
  
  export function ModelCard({ model, onAction, actionButton }: ModelCardProps) {
    // DUMB component - just render props
  }
  ```
- [ ] Use EXISTING atoms:
  - [ ] Import `Card` from `@rbee/ui/atoms/Card`
  - [ ] Import `Badge` from `@rbee/ui/atoms/Badge`
  - [ ] Import `Button` from `@rbee/ui/atoms/Button`
- [ ] Follow consistent card structure (see Phase 1.3)
- [ ] Create `index.ts`: `export { ModelCard } from './ModelCard'`
- [ ] Test with mock data

### 2.2 WorkerCard Organism

- [ ] Create directory: `rbee-ui/src/marketplace/organisms/WorkerCard/`
- [ ] Create `WorkerCard.tsx`:
  ```tsx
  // TEAM-400: Marketplace worker card organism
  import { Card } from '@rbee/ui/atoms/Card'
  import { Badge } from '@rbee/ui/atoms/Badge'
  import { Button } from '@rbee/ui/atoms/Button'
  
  export interface WorkerCardProps {
    worker: {
      id: string
      name: string
      description: string
      version: string
      platform: string[]
      architecture: string[]
      workerType: 'cpu' | 'cuda' | 'metal'
    }
    onAction?: (workerId: string) => void
    actionButton?: React.ReactNode
  }
  
  export function WorkerCard({ worker, onAction, actionButton }: WorkerCardProps) {
    // DUMB component - just render props
  }
  ```
- [ ] Use EXISTING atoms (Card, Badge, Button)
- [ ] Follow consistent card structure
- [ ] Create `index.ts`
- [ ] Test with mock data

### 2.3 MarketplaceGrid Organism

- [ ] Create directory: `rbee-ui/src/marketplace/organisms/MarketplaceGrid/`
- [ ] Create `MarketplaceGrid.tsx`:
  ```tsx
  // TEAM-400: Generic grid for marketplace items
  import { LoadingSpinner } from '@rbee/ui/atoms/LoadingSpinner'
  import { ErrorMessage } from '@rbee/ui/atoms/ErrorMessage'
  
  export interface MarketplaceGridProps<T> {
    items: T[]
    renderItem: (item: T) => React.ReactNode
    isLoading?: boolean
    error?: string
    emptyMessage?: string
  }
  
  export function MarketplaceGrid<T>({ items, renderItem, isLoading, error, emptyMessage }: MarketplaceGridProps<T>) {
    // Handle loading, error, empty states
    // Render responsive grid
  }
  ```
- [ ] Use EXISTING atoms (LoadingSpinner, ErrorMessage)
- [ ] Create `index.ts`
- [ ] Test all states

### 2.4 FilterBar Organism

- [ ] Create directory: `rbee-ui/src/marketplace/organisms/FilterBar/`
- [ ] Create `FilterBar.tsx`:
  ```tsx
  // TEAM-400: Filter controls for marketplace
  import { Input } from '@rbee/ui/atoms/Input'
  import { Select } from '@rbee/ui/atoms/Select'
  import { Button } from '@rbee/ui/atoms/Button'
  
  export interface FilterBarProps {
    search: string
    onSearchChange: (value: string) => void
    sort: string
    onSortChange: (value: string) => void
    sortOptions: Array<{ value: string; label: string }>
    onClearFilters: () => void
  }
  
  export function FilterBar(props: FilterBarProps) {
    // Search input + Sort dropdown + Clear button
  }
  ```
- [ ] Use EXISTING atoms (Input, Select, Button)
- [ ] Add debounce for search (300ms)
- [ ] Create `index.ts`
- [ ] Test filter changes

---

## 🎨 Phase 3: Templates - Page Sections (Days 3-4)

**TEAM-400:** Templates are reusable page sections. Follow commercial site pattern.

### 3.1 ModelListTemplate

- [ ] Create directory: `rbee-ui/src/marketplace/templates/ModelListTemplate/`
- [ ] Create `ModelListTemplateProps.tsx` (ALL data here for SSG):
  ```tsx
  // TEAM-400: Props for model list template
  import type { ModelCardProps } from '../../organisms/ModelCard'
  
  export interface ModelListTemplateProps {
    title: string
    description?: string
    models: ModelCardProps[]
    filters: {
      search: string
      sort: string
    }
    onFilterChange?: (filters: any) => void
  }
  ```
- [ ] Create `ModelListTemplate.tsx`:
  ```tsx
  // TEAM-400: DUMB template - just renders props
  import { ModelCard } from '../../organisms/ModelCard'
  import { MarketplaceGrid } from '../../organisms/MarketplaceGrid'
  import { FilterBar } from '../../organisms/FilterBar'
  import { TemplateContainer } from '@rbee/ui/molecules/TemplateContainer'
  
  export function ModelListTemplate(props: ModelListTemplateProps) {
    // Render FilterBar + MarketplaceGrid with ModelCards
  }
  ```
- [ ] Use TemplateContainer for section wrapping
- [ ] Create `index.ts`

### 3.2 ModelDetailTemplate

- [ ] Create directory: `rbee-ui/src/marketplace/templates/ModelDetailTemplate/`
- [ ] Create `ModelDetailTemplateProps.tsx`:
  ```tsx
  // TEAM-400: All model detail data (for SSG)
  export interface ModelDetailTemplateProps {
    model: {
      id: string
      name: string
      description: string
      author?: string
      // ... full model details
    }
    installButton?: React.ReactNode
    relatedModels?: Array<{...}>
  }
  ```
- [ ] Create `ModelDetailTemplate.tsx`:
  ```tsx
  // TEAM-400: DUMB template
  export function ModelDetailTemplate(props: ModelDetailTemplateProps) {
    // Render model details + install button + related models
  }
  ```
- [ ] Create `index.ts`

### 3.3 WorkerListTemplate

- [ ] Create directory: `rbee-ui/src/marketplace/templates/WorkerListTemplate/`
- [ ] Create `WorkerListTemplateProps.tsx`
- [ ] Create `WorkerListTemplate.tsx`:
  ```tsx
  // TEAM-400: Similar to ModelListTemplate but for workers
  import { WorkerCard } from '../../organisms/WorkerCard'
  import { MarketplaceGrid } from '../../organisms/MarketplaceGrid'
  ```
- [ ] Create `index.ts`

---

## 📄 Phase 4: Pages - Complete Pages (Day 5)

**TEAM-400:** Pages are DUMB - just render templates with props from Props file.

### 4.1 ModelsPage

- [ ] Create directory: `rbee-ui/src/marketplace/pages/ModelsPage/`
- [ ] Create `ModelsPageProps.tsx`:
  ```tsx
  // TEAM-400: ALL page data (perfect for SSG)
  import type { ModelListTemplateProps } from '../../templates/ModelListTemplate'
  
  export interface ModelsPageProps {
    seo: {
      title: string
      description: string
    }
    template: ModelListTemplateProps
  }
  
  // Example props for SSG
  export const defaultModelsPageProps: ModelsPageProps = {
    seo: {
      title: 'Browse AI Models | rbee Marketplace',
      description: 'Discover and download AI models...'
    },
    template: {
      title: 'AI Models',
      models: [],
      filters: { search: '', sort: 'popular' }
    }
  }
  ```
- [ ] Create `ModelsPage.tsx`:
  ```tsx
  // TEAM-400: DUMB page - just renders template
  import { ModelListTemplate } from '../../templates/ModelListTemplate'
  import type { ModelsPageProps } from './ModelsPageProps'
  
  export function ModelsPage({ template }: ModelsPageProps) {
    return <ModelListTemplate {...template} />
  }
  ```
- [ ] Create `index.ts`

### 4.2 ModelDetailPage

- [ ] Create directory: `rbee-ui/src/marketplace/pages/ModelDetailPage/`
- [ ] Create `ModelDetailPageProps.tsx`:
  ```tsx
  // TEAM-400: Full model detail data
  export interface ModelDetailPageProps {
    seo: { title: string; description: string }
    template: ModelDetailTemplateProps
  }
  ```
- [ ] Create `ModelDetailPage.tsx`:
  ```tsx
  // TEAM-400: DUMB page
  import { ModelDetailTemplate } from '../../templates/ModelDetailTemplate'
  
  export function ModelDetailPage({ template }: ModelDetailPageProps) {
    return <ModelDetailTemplate {...template} />
  }
  ```
- [ ] Create `index.ts`

### 4.3 WorkersPage

- [ ] Create directory: `rbee-ui/src/marketplace/pages/WorkersPage/`
- [ ] Create `WorkersPageProps.tsx`
- [ ] Create `WorkersPage.tsx`
- [ ] Create `index.ts`

## 📤 Phase 5: Export & Documentation (Day 6)

### 5.1 Update rbee-ui Exports

**TEAM-400:** Export marketplace components from rbee-ui package.

- [ ] Update `rbee-ui/src/index.ts` to include marketplace exports:
  ```typescript
  // TEAM-400: Marketplace components
  export * from './marketplace/organisms/ModelCard'
  export * from './marketplace/organisms/WorkerCard'
  export * from './marketplace/organisms/MarketplaceGrid'
  export * from './marketplace/organisms/FilterBar'
  export * from './marketplace/templates/ModelListTemplate'
  export * from './marketplace/templates/ModelDetailTemplate'
  export * from './marketplace/templates/WorkerListTemplate'
  export * from './marketplace/pages/ModelsPage'
  export * from './marketplace/pages/ModelDetailPage'
  export * from './marketplace/pages/WorkersPage'
  ```
- [ ] Build rbee-ui package: `cd frontend/packages/rbee-ui && pnpm build`
- [ ] Verify all exports work: `pnpm check-exports`

### 5.2 Create Marketplace README

- [ ] Create `rbee-ui/src/marketplace/README.md`:
  ```markdown
  # Marketplace Components
  
  **TEAM-400:** Marketplace-specific components for rbee.
  
  ## Structure
  
  ```
  marketplace/
  ├── organisms/   (Reusable cards and controls)
  ├── templates/   (Page sections)
  └── pages/       (Complete pages)
  ```
  
  ## Usage
  
  ### In Next.js (marketplace site)
  
  \`\`\`tsx
  import { ModelsPage, defaultModelsPageProps } from '@rbee/ui/marketplace/pages/ModelsPage'
  
  export default function Page() {
    return <ModelsPage {...defaultModelsPageProps} />
  }
  \`\`\`
  
  ### In Tauri (Keeper app)
  
  \`\`\`tsx
  import { ModelCard } from '@rbee/ui/marketplace/organisms/ModelCard'
  
  export function MarketplaceTab() {
    const models = useModels() // From marketplace SDK
    return models.map(model => <ModelCard key={model.id} model={model} />)
  }
  \`\`\`
  
  ## Principles
  
  1. **DUMB COMPONENTS** - No data fetching, only props
  2. **REUSE ATOMS/MOLECULES** - Don't recreate Button, Card, etc.
  3. **CONSISTENT STYLING** - Follow rbee-ui patterns
  4. **SSG-READY** - All data in Props files
  ```
- [ ] Add component examples with screenshots

### 5.3 Add Storybook Stories

**TEAM-400:** rbee-ui already has Storybook configured.

- [ ] Create `marketplace/organisms/ModelCard/ModelCard.stories.tsx`:
  ```tsx
  // TEAM-400: Storybook story for ModelCard
  import type { Meta, StoryObj } from '@storybook/react'
  import { ModelCard } from './ModelCard'
  
  const meta: Meta<typeof ModelCard> = {
    title: 'Marketplace/Organisms/ModelCard',
    component: ModelCard,
  }
  export default meta
  
  export const Default: StoryObj<typeof ModelCard> = {
    args: {
      model: {
        id: 'llama-3.2-1b',
        name: 'Llama 3.2 1B',
        description: 'Fast and efficient small language model',
        // ...
      }
    }
  }
  ```
- [ ] Create stories for:
  - [ ] `WorkerCard.stories.tsx`
  - [ ] `MarketplaceGrid.stories.tsx`
  - [ ] `FilterBar.stories.tsx`
  - [ ] `ModelListTemplate.stories.tsx`
- [ ] Run Storybook: `cd frontend/packages/rbee-ui && pnpm storybook`
- [ ] Verify all stories render correctly

---

## ✅ Phase 6: Testing (Day 7)

### 6.1 Component Tests

**TEAM-400:** Use existing rbee-ui test setup.

- [ ] Verify rbee-ui has testing dependencies
- [ ] Check `vitest.config.ts` exists
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
- [ ] Write tests for FilterBar:
  - [ ] Updates search on change
  - [ ] Updates sort on change
  - [ ] Clears filters correctly
  - [ ] Debounces search input
- [ ] Run tests: `pnpm test`
- [ ] Verify all tests pass

### 6.2 Integration Tests

**TEAM-400:** Test in actual Next.js marketplace app.

- [ ] Use marketplace app: `frontend/apps/marketplace/`
- [ ] Create test page: `app/test-components/page.tsx`
- [ ] Import and render:
  - [ ] `ModelsPage` with mock data
  - [ ] `ModelDetailPage` with mock data
  - [ ] `WorkersPage` with mock data
- [ ] Run dev server: `pnpm dev`
- [ ] Verify:
  - [ ] Components render correctly
  - [ ] Responsive layouts work
  - [ ] Loading states work
  - [ ] Error states work
- [ ] Build for production: `pnpm build`
- [ ] Verify SSG generates pages correctly

### 6.3 Visual Tests (Optional)

- [ ] Use Storybook for visual testing
- [ ] Review each story manually
- [ ] Check responsive layouts (mobile, tablet, desktop)
- [ ] Verify dark mode (if applicable)
- [ ] Take screenshots for documentation

---

## 📊 Success Criteria

### Must Have

- [ ] All marketplace components implemented:
  - [ ] 4 organisms (ModelCard, WorkerCard, MarketplaceGrid, FilterBar)
  - [ ] 3 templates (ModelList, ModelDetail, WorkerList)
  - [ ] 3 pages (ModelsPage, ModelDetailPage, WorkersPage)
- [ ] Components work in Next.js marketplace app
- [ ] rbee-ui package builds successfully
- [ ] All exports work from `@rbee/ui`
- [ ] README with examples
- [ ] Unit tests pass
- [ ] Components follow rbee-ui patterns (consistency!)

### Nice to Have

- [ ] Storybook stories for all components
- [ ] Integration tests in marketplace app
- [ ] Visual tests via Storybook
- [ ] >80% test coverage
- [ ] Accessibility tests (a11y)

---

## 🚀 Deliverables

1. **Components:** 10 marketplace components in `rbee-ui/src/marketplace/`
2. **Structure:** Follows atomic design (organisms → templates → pages)
3. **Exports:** All components exported from `@rbee/ui`
4. **Tests:** Unit tests for all components
5. **Documentation:** README with usage examples
6. **Storybook:** Interactive component gallery

---

## 📝 Notes

### Key Principles

1. **DUMB COMPONENTS** - No data fetching, only props
2. **REUSE ATOMS/MOLECULES** - Don't recreate existing components
3. **CONSISTENT** - Follow rbee-ui patterns (Card structure, spacing, etc.)
4. **SSG-READY** - All data in Props files
5. **TYPED** - Full TypeScript support
6. **TESTED** - Unit tests for all components

### Common Pitfalls

- ❌ Don't create separate package (use rbee-ui)
- ❌ Don't recreate atoms/molecules (Button, Card, Badge exist!)
- ❌ Don't mix card structures (follow existing pattern)
- ❌ Don't fetch data in components
- ✅ Use rbee-ui/src/marketplace/ directory
- ✅ Reuse existing atoms/molecules
- ✅ Follow commercial site pattern (Pages → Templates → Props)
- ✅ Accept data via props (perfect for SSG)

---

**Start with Phase 1, complete each checkbox in order!** ✅
