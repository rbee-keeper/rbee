import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms/Card'
import { Badge } from '@rbee/ui/atoms/Badge'
import { cn } from '@rbee/ui/utils'
import { TrendingDown, TrendingUp } from 'lucide-react'

export interface CostItem {
  /** Item label */
  label: string
  /** Cost value */
  value: string
  /** Is this a one-time cost? */
  oneTime?: boolean
  /** Highlight as savings */
  isSavings?: boolean
}

export interface BlogCostBreakdownProps {
  /** Card title */
  title: string
  /** Subtitle/description */
  subtitle?: string
  /** Before scenario description */
  before?: string
  /** After scenario description */
  after?: string
  /** Cost items */
  items: CostItem[]
  /** Summary items (e.g., monthly savings, annual savings, break-even) */
  summary?: CostItem[]
  /** Additional CSS classes */
  className?: string
}

/**
 * BlogCostBreakdown - ROI and cost analysis component for blog posts
 * Shows before/after scenarios with detailed cost breakdowns
 * 
 * @example
 * <BlogCostBreakdown
 *   title="SaaS Startup Use Case"
 *   before="Spending $800/month on OpenAI GPT-3.5"
 *   after="Deployed on 1× RTX 3090 ($800 hardware + €129 license)"
 *   items={[
 *     { label: "Hardware", value: "$800", oneTime: true },
 *     { label: "rbee License", value: "€129", oneTime: true },
 *     { label: "Power", value: "~$30/month" },
 *   ]}
 *   summary={[
 *     { label: "Monthly savings", value: "$770", isSavings: true },
 *     { label: "Annual savings", value: "$9,240", isSavings: true },
 *     { label: "Break-even", value: "1.2 months", isSavings: true },
 *   ]}
 * />
 */
export function BlogCostBreakdown({
  title,
  subtitle,
  before,
  after,
  items,
  summary,
  className,
}: BlogCostBreakdownProps) {
  return (
    <Card className={cn('border-green-500/20 bg-green-50/50 dark:bg-green-950/20', className)}>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-lg">{title}</CardTitle>
            {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
          </div>
          <Badge variant="outline" className="bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20">
            ROI Analysis
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Before/After */}
        {(before || after) && (
          <div className="space-y-2 text-sm">
            {before && (
              <div className="flex items-start gap-2">
                <TrendingUp className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <span className="font-semibold">Before rbee:</span> {before}
                </div>
              </div>
            )}
            {after && (
              <div className="flex items-start gap-2">
                <TrendingDown className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                <div>
                  <span className="font-semibold">After rbee:</span> {after}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Cost Items */}
        <div className="bg-background/50 rounded-lg p-4 space-y-2">
          <p className="font-semibold text-sm mb-3">Cost Breakdown:</p>
          <ul className="space-y-1.5 text-sm">
            {items.map((item, idx) => (
              <li key={idx} className="flex items-center justify-between">
                <span className="text-muted-foreground">
                  • {item.label}
                  {item.oneTime && (
                    <Badge variant="outline" className="ml-2 text-xs">
                      one-time
                    </Badge>
                  )}
                </span>
                <span className="font-medium">{item.value}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Summary */}
        {summary && summary.length > 0 && (
          <div className="space-y-1.5 text-sm pt-2 border-t border-border">
            {summary.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <span className="font-medium">{item.label}:</span>
                <span
                  className={cn(
                    'font-bold',
                    item.isSavings && 'text-green-600 dark:text-green-400',
                  )}
                >
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
