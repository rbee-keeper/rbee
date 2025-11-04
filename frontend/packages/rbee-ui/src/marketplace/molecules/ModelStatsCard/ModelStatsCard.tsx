// TEAM-405: Model statistics card
//! Reusable component for displaying model stats (downloads, likes, size)

import { Card, CardContent, CardHeader, CardTitle, Badge } from '@rbee/ui/atoms'
import { Download, Heart, HardDrive, LucideIcon } from 'lucide-react'

export interface ModelStat {
  icon: LucideIcon
  label: string
  value: string | number
  /** Render value as badge */
  badge?: boolean
  /** Badge variant */
  badgeVariant?: 'default' | 'secondary' | 'outline' | 'destructive'
}

export interface ModelStatsCardProps {
  stats: ModelStat[]
  title?: string
  className?: string
}

const defaultFormatNumber = (num: number | string): string => {
  if (typeof num === 'string') return num
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toString()
}

/**
 * Card for displaying model statistics
 * 
 * @example
 * ```tsx
 * <ModelStatsCard
 *   title="Statistics"
 *   stats={[
 *     { icon: Download, label: 'Downloads', value: 151 },
 *     { icon: Heart, label: 'Likes', value: 0 },
 *     { icon: HardDrive, label: 'Size', value: '4.4 GB', badge: true }
 *   ]}
 * />
 * ```
 */
export function ModelStatsCard({ stats, title = 'Statistics', className }: ModelStatsCardProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          const formattedValue = typeof stat.value === 'number' 
            ? defaultFormatNumber(stat.value) 
            : stat.value

          return (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Icon className="size-4" />
                <span>{stat.label}</span>
              </div>
              {stat.badge ? (
                <Badge variant={stat.badgeVariant || 'outline'} className="font-mono">
                  {formattedValue}
                </Badge>
              ) : (
                <span className="font-semibold tabular-nums">{formattedValue}</span>
              )}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

// Export common icons for convenience
export { Download, Heart, HardDrive }
