// TEAM-405: Model table presentation component
//! Pure presentational component for displaying models in a table
//! Works with any data source (Tauri, SSG, API)

import { Badge } from '@rbee/ui/atoms'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@rbee/ui/atoms'
import { Download, Heart } from 'lucide-react'

export interface ModelTableItem {
  id: string
  name: string
  description: string
  author?: string | null
  downloads: number
  likes: number
  tags: string[]
}

export interface ModelTableProps {
  /** Models to display */
  models: ModelTableItem[]
  
  /** Called when a model row is clicked */
  onModelClick?: (modelId: string) => void
  
  /** Format number for display (e.g., 1000 -> "1K") */
  formatNumber?: (num: number) => string
  
  /** Loading state */
  isLoading?: boolean
  
  /** Error message */
  error?: string
  
  /** Empty state message */
  emptyMessage?: string
  
  /** Empty state description */
  emptyDescription?: string
}

const defaultFormatNumber = (num: number): string => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toString()
}

export function ModelTable({
  models,
  onModelClick,
  formatNumber = defaultFormatNumber,
  isLoading = false,
  error,
  emptyMessage = 'No models found',
  emptyDescription = 'Try adjusting your search query',
}: ModelTableProps) {
  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(10)].map((_, i) => (
          <div key={i} className="h-14 rounded-md border border-border/50 bg-muted/20 animate-pulse" />
        ))}
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-destructive">Error: {error}</p>
      </div>
    )
  }

  // Empty state
  if (models.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground mb-2">{emptyMessage}</p>
        <p className="text-sm text-muted-foreground">{emptyDescription}</p>
      </div>
    )
  }

  // Table
  return (
    <div className="rounded-lg border border-border/50 overflow-x-auto">
      <Table className="table-fixed">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[40%]">Model</TableHead>
            <TableHead className="w-[12%]">Author</TableHead>
            <TableHead className="w-[12%] text-right">
              <div className="flex items-center justify-end gap-1">
                <Download className="size-3" />
                <span className="hidden sm:inline">DL</span>
              </div>
            </TableHead>
            <TableHead className="w-[12%] text-right">
              <div className="flex items-center justify-end gap-1">
                <Heart className="size-3" />
                <span className="hidden sm:inline">Likes</span>
              </div>
            </TableHead>
            <TableHead className="w-[24%]">Tags</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => (
            <TableRow
              key={model.id}
              className="cursor-pointer"
              onClick={() => onModelClick?.(model.id)}
            >
              <TableCell>
                <div className="min-w-0">
                  <div className="font-semibold text-foreground truncate">
                    {model.name}
                  </div>
                  <div className="text-xs text-muted-foreground truncate mt-0.5">
                    {model.description}
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <span className="text-sm text-muted-foreground truncate block">
                  {model.author || '—'}
                </span>
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {formatNumber(model.downloads)}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {formatNumber(model.likes)}
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1">
                  {model.tags.slice(0, 2).map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {model.tags.length > 2 && (
                    <Badge variant="outline" className="text-xs">
                      +{model.tags.length - 2}
                    </Badge>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
