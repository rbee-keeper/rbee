// TEAM-481: HuggingFace Model List Card - reusable component for model listings
// Used in homepage and HuggingFace models page
// TEAM-505: Refactored to use existing rbee-ui components (Card, Badge, lucide-react icons)
// TEAM-505: Complete redesign - compact, visually appealing, better space utilization

'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Card } from '@rbee/ui/atoms/Card'
import { cn } from '@rbee/ui/utils'
import { Download, Heart } from 'lucide-react'
import Link from 'next/link'

export interface HFModelListCardProps {
  model: {
    id: string
    name: string
    author: string
    type: string
    downloads: number
    likes: number
  }
  href?: string
  className?: string
}

export function HFModelListCard({ model, href, className }: HFModelListCardProps) {
  // TEAM-481: Format large numbers for brevity (23M instead of 23,000,000)
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const content = (
    <Card
      className={cn(
        'group relative overflow-hidden border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 h-full',
        href && 'cursor-pointer hover:-translate-y-0.5',
        className,
      )}
    >
      {/* Compact single-section layout */}
      <div className="p-4 space-y-3">
        {/* Header: Model name + Type badge */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-medium leading-tight truncate">
              <span className="text-muted-foreground">{model.author}</span>
              <span className="text-foreground">/</span>
              <span className="font-semibold text-foreground">{model.name.split('/').pop() || model.name}</span>
            </h3>
          </div>
          <Badge variant="secondary" className="font-mono text-[10px] px-2 py-0.5 shrink-0">
            {model.type}
          </Badge>
        </div>

        {/* Stats row - compact with better spacing */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5 group-hover:text-foreground transition-colors">
            <Download className="w-3.5 h-3.5" aria-hidden="true" />
            <span className="font-medium tabular-nums">{formatNumber(model.downloads)}</span>
          </div>
          <div className="flex items-center gap-1.5 group-hover:text-foreground transition-colors">
            <Heart className="w-3.5 h-3.5" aria-hidden="true" />
            <span className="font-medium tabular-nums">{formatNumber(model.likes)}</span>
          </div>
        </div>
      </div>

      {/* Subtle gradient accent on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/0 group-hover:from-primary/5 group-hover:via-transparent group-hover:to-transparent transition-all duration-300 pointer-events-none" />
    </Card>
  )

  if (href) {
    return (
      <Link href={href} className="block h-full">
        {content}
      </Link>
    )
  }

  return content
}
