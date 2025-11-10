import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms/Card'
import { Badge } from '@rbee/ui/atoms/Badge'
import { cn } from '@rbee/ui/utils'
import { Check, X } from 'lucide-react'

export interface ComparisonRow {
  /** Feature/criterion name */
  feature: string
  /** Values for each column (string, boolean, or React node) */
  values: (string | boolean | React.ReactNode)[]
}

export interface BlogComparisonTableProps {
  /** Table title */
  title?: string
  /** Column headers */
  columns: string[]
  /** Highlight column index (0-based) */
  highlightColumn?: number
  /** Comparison rows */
  rows: ComparisonRow[]
  /** Additional CSS classes */
  className?: string
}

/**
 * BlogComparisonTable - Specialized comparison table for blog posts
 * Supports highlighting a specific column (e.g., "rbee" vs competitors)
 * 
 * @example
 * <BlogComparisonTable
 *   title="Deployment Comparison"
 *   columns={["rbee (SSH)", "Kubernetes", "Docker Swarm"]}
 *   highlightColumn={0}
 *   rows={[
 *     { feature: "Setup Time", values: ["5 minutes", "2-6 months", "1-2 weeks"] },
 *     { feature: "Complexity", values: ["Low", "Very High", "Medium"] },
 *   ]}
 * />
 */
export function BlogComparisonTable({
  title,
  columns,
  highlightColumn,
  rows,
  className,
}: BlogComparisonTableProps) {
  const renderCell = (value: string | boolean | React.ReactNode) => {
    if (typeof value === 'boolean') {
      return value ? (
        <Check className="h-5 w-5 text-green-600 dark:text-green-400 mx-auto" />
      ) : (
        <X className="h-5 w-5 text-red-600 dark:text-red-400 mx-auto" />
      )
    }
    return value
  }

  return (
    <Card className={cn('overflow-hidden', className)}>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="px-4 py-3 text-left text-sm font-semibold">Feature</th>
                {columns.map((col, idx) => (
                  <th
                    key={idx}
                    className={cn(
                      'px-4 py-3 text-left text-sm font-semibold',
                      highlightColumn === idx && 'bg-primary/10',
                    )}
                  >
                    <div className="flex items-center gap-2">
                      {col}
                      {highlightColumn === idx && (
                        <Badge variant="default" className="text-xs">
                          Recommended
                        </Badge>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIdx) => (
                <tr
                  key={rowIdx}
                  className={cn(
                    'border-b border-border transition-colors hover:bg-muted/30',
                    rowIdx % 2 === 0 && 'bg-muted/10',
                  )}
                >
                  <td className="px-4 py-3 text-sm font-medium">{row.feature}</td>
                  {row.values.map((value, colIdx) => (
                    <td
                      key={colIdx}
                      className={cn(
                        'px-4 py-3 text-sm',
                        highlightColumn === colIdx && 'bg-primary/5 font-medium',
                      )}
                    >
                      {renderCell(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
