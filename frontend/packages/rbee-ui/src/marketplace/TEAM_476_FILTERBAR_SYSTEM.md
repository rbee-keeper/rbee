# TEAM-476: Comprehensive FilterBar System

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Complete filter system for HuggingFace and CivitAI with ALL filter options

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ FilterBar (Organism)                                            │
│                                                                 │
│ ┌────────────────────────────────┐  ┌──────────────────────┐  │
│ │ LEFT: Filters (Grid)           │  │ RIGHT: Sort + Clear  │  │
│ │                                │  │                      │  │
│ │ [FilterSearch]                 │  │ [SortDropdown]       │  │
│ │ [FilterDropdown]               │  │ [Clear Button]       │  │
│ │ [FilterMultiSelect]            │  │                      │  │
│ │ [FilterMultiSelect]            │  │                      │  │
│ └────────────────────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components Created

### 1. SortDropdown (Molecule) - RIGHT SIDE ✅

**Location:** `/molecules/SortDropdown/`

**Purpose:** Single dropdown for sorting with icon

```typescript
<SortDropdown
  value="downloads"
  onChange={(value) => setSort(value)}
  options={[
    { value: 'downloads', label: 'Most Downloaded' },
    { value: 'likes', label: 'Most Liked' },
    { value: 'trending', label: 'Trending' },
  ]}
/>
```

**Features:**
- Icon changes based on selection (ArrowDownAZ / ArrowUpAZ)
- Clean dropdown UI
- Type-safe options

### 2. FilterSearch (Molecule) - LEFT SIDE ✅

**Location:** `/molecules/FilterSearch/`

**Purpose:** Search input with debounce and clear button

```typescript
<FilterSearch
  label="Search"
  value={search}
  onChange={(value) => setSearch(value)}
  placeholder="Search models..."
  debounceMs={300}
/>
```

**Features:**
- Debounced input (300ms default)
- Clear button (X icon)
- Search icon
- Label support

### 3. FilterDropdown (Molecule) - LEFT SIDE ✅

**Location:** `/molecules/FilterDropdown/`

**Purpose:** Single-select dropdown filter

```typescript
<FilterDropdown
  label="Library"
  value={library}
  onChange={(value) => setLibrary(value)}
  options={[
    { value: 'transformers', label: 'Transformers' },
    { value: 'diffusers', label: 'Diffusers' },
    { value: 'pytorch', label: 'PyTorch' },
  ]}
  allowClear={true}
/>
```

**Features:**
- Clear selection option
- Label support
- Placeholder
- Type-safe options

### 4. FilterMultiSelect (Molecule) - LEFT SIDE ✅

**Location:** `/molecules/FilterMultiSelect/`

**Purpose:** Multi-select filter with checkboxes

```typescript
<FilterMultiSelect
  label="Model Types"
  values={types}
  onChange={(values) => setTypes(values)}
  options={[
    { value: 'Checkpoint', label: 'Checkpoint' },
    { value: 'LORA', label: 'LORA' },
    { value: 'ControlNet', label: 'ControlNet' },
  ]}
  maxDisplay={3}
/>
```

**Features:**
- Checkbox list in popover
- Badge display for selected items
- Clear all button
- Max display count with "+N" badge
- Scrollable list

### 5. FilterBar (Organism) - MAIN CONTAINER ✅

**Location:** `/organisms/FilterBar/`

**Purpose:** Main container with LEFT filters and RIGHT sort

```typescript
<FilterBar
  filters={
    <>
      <FilterSearch label="Search" value={search} onChange={setSearch} />
      <FilterDropdown label="Library" value={library} onChange={setLibrary} options={libraryOptions} />
      <FilterMultiSelect label="Types" values={types} onChange={setTypes} options={typeOptions} />
    </>
  }
  sort={sort}
  onSortChange={setSort}
  sortOptions={sortOptions}
  hasActiveFilters={hasFilters}
  onClearFilters={clearAll}
/>
```

**Features:**
- Responsive grid layout (1-4 columns)
- Filters on LEFT
- Sort + Clear on RIGHT
- Flexible filter composition

## Usage Examples

### HuggingFace Filters

```typescript
import { FilterBar, FilterSearch, FilterDropdown, FilterMultiSelect } from '@rbee/ui/marketplace'

function HuggingFaceFilters() {
  const [search, setSearch] = useState('')
  const [library, setLibrary] = useState<string>()
  const [pipelineTag, setPipelineTag] = useState<string>()
  const [language, setLanguage] = useState<string>()
  const [sort, setSort] = useState('downloads')

  return (
    <FilterBar
      filters={
        <>
          <FilterSearch
            label="Search"
            value={search}
            onChange={setSearch}
            placeholder="Search models..."
          />
          <FilterDropdown
            label="Library"
            value={library}
            onChange={setLibrary}
            options={[
              { value: 'transformers', label: 'Transformers' },
              { value: 'diffusers', label: 'Diffusers' },
              { value: 'pytorch', label: 'PyTorch' },
            ]}
          />
          <FilterDropdown
            label="Task"
            value={pipelineTag}
            onChange={setPipelineTag}
            options={[
              { value: 'text-generation', label: 'Text Generation' },
              { value: 'image-classification', label: 'Image Classification' },
            ]}
          />
          <FilterDropdown
            label="Language"
            value={language}
            onChange={setLanguage}
            options={[
              { value: 'en', label: 'English' },
              { value: 'fr', label: 'French' },
            ]}
          />
        </>
      }
      sort={sort}
      onSortChange={setSort}
      sortOptions={[
        { value: 'downloads', label: 'Most Downloaded' },
        { value: 'likes', label: 'Most Liked' },
        { value: 'trending', label: 'Trending' },
        { value: 'updated', label: 'Recently Updated' },
        { value: 'created', label: 'Recently Created' },
      ]}
      hasActiveFilters={search !== '' || library || pipelineTag || language}
      onClearFilters={() => {
        setSearch('')
        setLibrary(undefined)
        setPipelineTag(undefined)
        setLanguage(undefined)
      }}
    />
  )
}
```

### CivitAI Filters

```typescript
import { FilterBar, FilterSearch, FilterDropdown, FilterMultiSelect } from '@rbee/ui/marketplace'

function CivitAIFilters() {
  const [query, setQuery] = useState('')
  const [types, setTypes] = useState<string[]>([])
  const [baseModels, setBaseModels] = useState<string[]>([])
  const [nsfwLevel, setNsfwLevel] = useState<string[]>([])
  const [sort, setSort] = useState('Most Downloaded')
  const [period, setPeriod] = useState('AllTime')

  return (
    <FilterBar
      filters={
        <>
          <FilterSearch
            label="Search"
            value={query}
            onChange={setQuery}
            placeholder="Search models..."
          />
          <FilterMultiSelect
            label="Model Types"
            values={types}
            onChange={setTypes}
            options={[
              { value: 'Checkpoint', label: 'Checkpoint' },
              { value: 'LORA', label: 'LORA' },
              { value: 'ControlNet', label: 'ControlNet' },
              { value: 'TextualInversion', label: 'Textual Inversion' },
            ]}
          />
          <FilterMultiSelect
            label="Base Models"
            values={baseModels}
            onChange={setBaseModels}
            options={[
              { value: 'SD 1.5', label: 'SD 1.5' },
              { value: 'SDXL 1.0', label: 'SDXL 1.0' },
              { value: 'Flux.1 D', label: 'Flux.1 D' },
            ]}
          />
          <FilterMultiSelect
            label="NSFW Level"
            values={nsfwLevel}
            onChange={setNsfwLevel}
            options={[
              { value: '1', label: 'None' },
              { value: '2', label: 'Soft' },
              { value: '4', label: 'Mature' },
            ]}
          />
          <FilterDropdown
            label="Time Period"
            value={period}
            onChange={setPeriod}
            options={[
              { value: 'AllTime', label: 'All Time' },
              { value: 'Month', label: 'Past Month' },
              { value: 'Week', label: 'Past Week' },
              { value: 'Day', label: 'Today' },
            ]}
          />
        </>
      }
      sort={sort}
      onSortChange={setSort}
      sortOptions={[
        { value: 'Most Downloaded', label: 'Most Downloaded' },
        { value: 'Highest Rated', label: 'Highest Rated' },
        { value: 'Newest', label: 'Newest' },
      ]}
      hasActiveFilters={query !== '' || types.length > 0 || baseModels.length > 0}
      onClearFilters={() => {
        setQuery('')
        setTypes([])
        setBaseModels([])
        setNsfwLevel([])
        setPeriod('AllTime')
      }}
    />
  )
}
```

## Responsive Layout

**Desktop (lg+):**
```
┌────────────────────────────────────────────────────────────┐
│ [Filter1] [Filter2] [Filter3] [Filter4]    [Sort] [Clear] │
└────────────────────────────────────────────────────────────┘
```

**Tablet (md):**
```
┌────────────────────────────────────────────────────────────┐
│ [Filter1] [Filter2]                                        │
│ [Filter3] [Filter4]                        [Sort] [Clear]  │
└────────────────────────────────────────────────────────────┘
```

**Mobile:**
```
┌────────────────────────────────────────────────────────────┐
│ [Filter1]                                                  │
│ [Filter2]                                                  │
│ [Filter3]                                                  │
│ [Filter4]                                                  │
│ [Sort] [Clear]                                             │
└────────────────────────────────────────────────────────────┘
```

## Grid Breakpoints

- **Mobile:** 1 column
- **md:** 2 columns
- **lg:** 3 columns
- **xl:** 4 columns

## Components Summary

| Component | Type | Purpose | Features |
|-----------|------|---------|----------|
| **SortDropdown** | Molecule | Sort selection | Icon, dropdown, type-safe |
| **FilterSearch** | Molecule | Text search | Debounce, clear, icon |
| **FilterDropdown** | Molecule | Single select | Clear, label, placeholder |
| **FilterMultiSelect** | Molecule | Multi select | Checkboxes, badges, popover |
| **FilterBar** | Organism | Main container | Grid layout, responsive |

## File Structure

```
/packages/rbee-ui/src/marketplace/
├── molecules/
│   ├── SortDropdown/
│   │   ├── SortDropdown.tsx
│   │   └── index.ts
│   ├── FilterSearch/
│   │   ├── FilterSearch.tsx
│   │   └── index.ts
│   ├── FilterDropdown/
│   │   ├── FilterDropdown.tsx
│   │   └── index.ts
│   └── FilterMultiSelect/
│       ├── FilterMultiSelect.tsx
│       └── index.ts
└── organisms/
    └── FilterBar/
        ├── FilterBar.tsx
        └── index.ts
```

## Next Steps

1. ✅ Components created
2. ⏭️ Export from main index
3. ⏭️ Use in HuggingFace page
4. ⏭️ Use in CivitAI page
5. ⏭️ Add Storybook stories

---

**TEAM-476 RULE ZERO:** Filters LEFT, Sort RIGHT. All vendor filter options supported. Responsive grid layout. Type-safe components!
