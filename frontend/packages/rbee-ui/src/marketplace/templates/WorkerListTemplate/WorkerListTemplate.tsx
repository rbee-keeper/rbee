// TEAM-401: Similar to ModelListTemplate but for workers
import { FilterBar } from '../../organisms/FilterBar'
import { MarketplaceGrid } from '../../organisms/MarketplaceGrid'
import { WorkerCard } from '../../organisms/WorkerCard'
import { defaultWorkerSortOptions, type WorkerListTemplateProps } from './WorkerListTemplateProps'

export function WorkerListTemplate({
  title,
  description,
  workers,
  filters = { search: '', sort: 'name' },
  sortOptions = defaultWorkerSortOptions,
  onFilterChange,
  onWorkerAction,
  onWorkerClick,
  isLoading,
  error,
  emptyMessage = 'No workers found',
  emptyDescription = 'Try adjusting your search or filters',
}: WorkerListTemplateProps) {
  const handleSearchChange = (search: string) => {
    onFilterChange?.({ ...filters, search })
  }

  const handleSortChange = (sort: string) => {
    onFilterChange?.({ ...filters, sort })
  }

  const handleClearFilters = () => {
    onFilterChange?.({ search: '', sort: sortOptions[0]?.value || 'name' })
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
          sort={filters.sort}
          onSortChange={handleSortChange}
          sortOptions={sortOptions}
          onClearFilters={handleClearFilters}
        />
      )}

      {/* Grid */}
      {isLoading ? (
        <div className="text-center py-12">Loading...</div>
      ) : error ? (
        <div className="text-center py-12 text-destructive">{error}</div>
      ) : workers.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-lg font-medium">{emptyMessage}</p>
          <p className="text-muted-foreground mt-2">{emptyDescription}</p>
        </div>
      ) : (
        <MarketplaceGrid>
          {workers.map((worker) => (
            <WorkerCard
              key={worker.id}
              worker={worker}
              {...(onWorkerAction ? { onAction: onWorkerAction } : {})}
              {...(onWorkerClick ? { onClick: onWorkerClick } : {})}
            />
          ))}
        </MarketplaceGrid>
      )}
    </div>
  )
}
