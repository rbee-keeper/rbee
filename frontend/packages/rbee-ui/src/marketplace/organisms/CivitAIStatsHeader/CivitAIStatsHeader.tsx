// TEAM-463: Premium stats header for CivitAI models
// Beautiful animated stat cards with icons

'use client'

import { Card } from '@rbee/ui/atoms'
import { Download, Heart, Star } from 'lucide-react'
import { cn } from '@rbee/ui/utils'

export interface CivitAIStatsHeaderProps {
  downloads: number
  likes: number
  rating: number
  className?: string
}

export function CivitAIStatsHeader({ downloads, likes, rating, className }: CivitAIStatsHeaderProps) {
  // TEAM-464: Show "N/A" for unrated models (rating === 0 or null/undefined)
  const ratingDisplay = rating > 0 ? rating.toFixed(1) : 'N/A'
  
  const stats = [
    {
      icon: Download,
      value: downloads.toLocaleString(),
      label: 'Downloads',
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
    },
    {
      icon: Heart,
      value: likes.toLocaleString(),
      label: 'Likes',
      color: 'text-pink-500',
      bgColor: 'bg-pink-500/10',
    },
    {
      icon: Star,
      value: ratingDisplay,
      label: 'Rating',
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
    },
  ]

  return (
    <div className={cn('grid grid-cols-3 gap-6', className)}>
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <Card
            key={stat.label}
            className="group relative overflow-hidden p-6 transition-all duration-300 hover:shadow-lg hover:scale-105"
          >
            {/* Background Gradient */}
            <div className={cn('absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100', stat.bgColor)} />
            
            {/* Content */}
            <div className="relative flex items-center gap-4">
              <div className={cn('rounded-xl p-3 transition-colors', stat.bgColor, 'group-hover:scale-110 transition-transform duration-300')}>
                <Icon className={cn('size-6', stat.color)} />
              </div>
              <div>
                <div className="text-2xl font-bold tabular-nums">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            </div>

            {/* Hover Effect Line */}
            <div className={cn('absolute bottom-0 left-0 h-1 w-0 transition-all duration-300 group-hover:w-full', stat.color.replace('text-', 'bg-'))} />
          </Card>
        )
      })}
    </div>
  )
}
