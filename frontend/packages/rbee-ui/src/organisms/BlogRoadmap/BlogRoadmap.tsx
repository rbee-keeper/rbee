import { Card, CardContent } from '@rbee/ui/atoms/Card'
import { Badge } from '@rbee/ui/atoms/Badge'
import { cn } from '@rbee/ui/utils'
import { Calendar, CheckCircle, Clock } from 'lucide-react'

export interface RoadmapItem {
  /** Milestone name (e.g., "M1", "Q1 2026") */
  milestone: string
  /** Phase title */
  title: string
  /** Phase description */
  description: string
  /** Status */
  status?: 'completed' | 'in-progress' | 'planned'
  /** Icon emoji */
  emoji?: string
}

export interface BlogRoadmapProps {
  /** Roadmap items */
  items: RoadmapItem[]
  /** Additional CSS classes */
  className?: string
}

/**
 * BlogRoadmap - Product roadmap timeline for blog posts
 * Shows milestones with visual progress indicators
 * 
 * @example
 * <BlogRoadmap
 *   items={[
 *     {
 *       milestone: "M1 (Q1 2026)",
 *       title: "Foundation",
 *       description: "Core orchestration, chat models, basic GUI",
 *       status: "in-progress",
 *       emoji: "🎯"
 *     },
 *     {
 *       milestone: "M2 (Q2 2026)",
 *       title: "Expansion",
 *       description: "Image generation, TTS, premium modules",
 *       status: "planned",
 *       emoji: "🎨"
 *     }
 *   ]}
 * />
 */
export function BlogRoadmap({ items, className }: BlogRoadmapProps) {
  const getStatusConfig = (status?: string) => {
    switch (status) {
      case 'completed':
        return {
          icon: CheckCircle,
          color: 'text-green-600 dark:text-green-400',
          bgColor: 'bg-green-500/10 border-green-500/30',
          badge: 'Completed',
          badgeVariant: 'default' as const,
        }
      case 'in-progress':
        return {
          icon: Clock,
          color: 'text-blue-600 dark:text-blue-400',
          bgColor: 'bg-blue-500/10 border-blue-500/30',
          badge: 'In Progress',
          badgeVariant: 'default' as const,
        }
      default:
        return {
          icon: Calendar,
          color: 'text-muted-foreground',
          bgColor: 'bg-muted/50 border-muted',
          badge: 'Planned',
          badgeVariant: 'outline' as const,
        }
    }
  }

  return (
    <div className={cn('space-y-4 my-6', className)}>
      {items.map((item, idx) => {
        const config = getStatusConfig(item.status)
        const Icon = config.icon
        const opacity = item.status === 'planned' ? 'opacity-70' : 'opacity-100'

        return (
          <Card
            key={idx}
            className={cn(
              'border-l-4 transition-all hover:shadow-md',
              config.bgColor,
              opacity,
            )}
          >
            <CardContent className="p-4 sm:p-5">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  {item.emoji && <span className="text-lg">{item.emoji}</span>}
                  <h3 className="font-semibold text-base">
                    {item.milestone} - {item.title}
                  </h3>
                </div>
                <Badge variant={config.badgeVariant} className="flex-shrink-0">
                  <Icon className="h-3 w-3 mr-1" />
                  {config.badge}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground pl-7">{item.description}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
