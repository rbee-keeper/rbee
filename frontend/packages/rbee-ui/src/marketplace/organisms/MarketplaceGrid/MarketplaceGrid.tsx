// TEAM-401: Generic grid for marketplace items
import { Alert } from '@rbee/ui/atoms/Alert'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@rbee/ui/atoms/Empty'
import { Spinner } from '@rbee/ui/atoms/Spinner'
import { PackageOpen } from 'lucide-react'
import * as React from 'react'

export interface MarketplaceGridProps<T> {
  items: T[]
  renderItem: (item: T, index: number) => React.ReactNode
  isLoading?: boolean
  error?: string
  emptyMessage?: string
  emptyDescription?: string
  columns?: 1 | 2 | 3 | 4
  pagination?: React.ReactNode
}

export function MarketplaceGrid<T>({
  items,
  renderItem,
  isLoading = false,
  error,
  emptyMessage = 'No items found',
  emptyDescription = 'Try adjusting your filters or search query',
  columns = 3,
  pagination,
}: MarketplaceGridProps<T>) {
  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex flex-col items-center gap-3">
          <Spinner className="size-8" />
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="py-8">
        <Alert variant="destructive">
          <p className="font-medium">Error loading items</p>
          <p className="text-sm mt-1">{error}</p>
        </Alert>
      </div>
    )
  }

  // Empty state
  if (items.length === 0) {
    return (
      <div className="py-12">
        <Empty>
          <EmptyHeader>
            <EmptyMedia>
              <PackageOpen className="size-12" />
            </EmptyMedia>
            <EmptyTitle>{emptyMessage}</EmptyTitle>
            <EmptyDescription>{emptyDescription}</EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    )
  }

  // Grid layout
  const gridColsClass = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  }[columns]

  return (
    <div className="space-y-6">
      <div className={`grid ${gridColsClass} gap-6`}>
        {items.map((item, index) => (
          <div key={index}>{renderItem(item, index)}</div>
        ))}
      </div>
      
      {pagination && <div className="flex justify-center">{pagination}</div>}
    </div>
  )
}
