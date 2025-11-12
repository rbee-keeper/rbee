// TEAM-481: HuggingFace Model List Card - reusable component for model listings
// Used in homepage and HuggingFace models page

'use client'

import { cn } from '@rbee/ui/utils'
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
    <div
      className={cn(
        'border border-border rounded-lg p-5 hover:border-primary/50 transition-colors bg-card h-full min-h-[120px] flex flex-col',
        href && 'cursor-pointer',
        className,
      )}
    >
      {/* Model name */}
      <div className="mb-4 flex-grow">
        <h3 className="text-lg truncate">
          <span className="font-light text-muted-foreground">{model.author}</span>/
          <span className="font-bold">{model.name.split('/').pop() || model.name}</span>
        </h3>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-sm text-muted-foreground flex-wrap mt-auto">
        <div className="flex items-center gap-1.5">
          <span className="text-xs bg-muted px-2 py-1 rounded font-mono">{model.type}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <span className="font-medium">{formatNumber(model.downloads)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
              clipRule="evenodd"
            />
          </svg>
          <span className="font-medium">{formatNumber(model.likes)}</span>
        </div>
      </div>
    </div>
  )

  if (href) {
    return (
      <Link href={href} className="block">
        {content}
      </Link>
    )
  }

  return content
}
