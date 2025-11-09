// TEAM-401: DUMB template - just renders props
import { FilterBar } from '../../organisms/FilterBar'
import { MarketplaceGrid } from '../../organisms/MarketplaceGrid'
import { ModelCard } from '../../organisms/ModelCard'
import { defaultModelSortOptions, type ModelListTemplateProps } from './ModelListTemplateProps'

export function ModelListTemplate({
  title,
  description,
  models,
  filters = { search: '', sort: 'popular' },
  sortOptions = defaultModelSortOptions,
  onFilterChange,
  onModelAction,
  isLoading,
  error,
  emptyMessage = 'No models found',
  emptyDescription = 'Try adjusting your search or filters',
}: ModelListTemplateProps) {
  const handleSearchChange = (search: string) => {
    onFilterChange?.({ ...filters, search })
  }

  const handleSortChange = (sort: string) => {
    onFilterChange?.({ ...filters, sort })
  }

  const handleClearFilters = () => {
    onFilterChange?.({ search: '', sort: sortOptions[0]?.value || 'popular' })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-serif font-bold tracking-tight">{title}</h1>
        {description && <p className="text-muted-foreground text-lg">{description}</p>}
      </div>

      {/* Filters */}
      {onFilterChange && (
        <FilterBar
          search={filters.search}
          onSearchChange={handleSearchChange}
          sort={filters.sort}
          onSortChange={handleSortChange}
          sortOptions={sortOptions}
          onClearFilters={handleClearFilters}
        />
      )}

      {/* Grid */}
      <MarketplaceGrid
        items={models}
        renderItem={(model) => <ModelCard key={model.id} model={model} onAction={onModelAction} />}
        isLoading={isLoading}
        error={error}
        emptyMessage={emptyMessage}
        emptyDescription={emptyDescription}
        columns={3}
      />
    </div>
  )
}
