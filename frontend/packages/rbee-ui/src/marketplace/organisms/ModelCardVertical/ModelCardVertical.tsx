// TEAM-422: Vertical model card for CivitAI-style portrait images
// TEAM-461: Added automatic image fallback handling
// TEAM-479: Added href prop to make cards clickable
import { Badge } from '@rbee/ui/atoms/Badge'
import { Card, CardFooter } from '@rbee/ui/atoms/Card'
import { Download, Heart, Sparkles, User } from 'lucide-react'
import Link from 'next/link'
import { ImageWithFallback } from './ImageWithFallback'

export interface ModelCardVerticalProps {
  model: {
    id: string
    name: string
    description: string
    author?: string
    imageUrl?: string
    fallbackImages?: string[]
    tags: string[]
    downloads: number
    likes: number
    size: string
  }
  href?: string
}

export function ModelCardVertical({ model, href }: ModelCardVerticalProps) {
  // TEAM-422: Pure SSG component - no useState
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const cardContent = (
    <Card className="group h-full flex flex-col overflow-hidden border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-xl hover:shadow-primary/5 hover:-translate-y-1 cursor-pointer">
      {/* Vertical Image Section - Portrait Aspect Ratio */}
      <div className="relative w-full aspect-[3/4] overflow-hidden bg-gradient-to-br from-primary/10 via-background to-muted">
        {model.imageUrl ? (
          <>
            <ImageWithFallback
              src={model.imageUrl}
              alt={model.name}
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              {...(model.fallbackImages ? { fallbackImages: model.fallbackImages } : {})}
            />
            {/* Gradient overlay for better text readability */}
            <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent" />
          </>
        ) : (
          // Fallback gradient when no image
          <div className="w-full h-full flex items-center justify-center">
            <Sparkles className="size-16 text-primary/30" />
          </div>
        )}

        {/* Top badges overlay */}
        <div className="absolute top-3 left-3 right-3 flex items-start justify-between gap-2">
          <div className="flex flex-wrap gap-1.5">
            {model.tags.slice(0, 2).map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="text-xs font-medium bg-black/40 backdrop-blur-sm text-white border-white/20"
              >
                {tag}
              </Badge>
            ))}
          </div>
        </div>

        {/* Bottom stats overlay */}
        <div className="absolute bottom-0 left-0 right-0 p-4">
          <div className="flex items-center gap-3 text-xs font-medium text-white drop-shadow-lg mb-3">
            <div className="flex items-center gap-1.5 bg-black/40 backdrop-blur-sm rounded-full px-3 py-1.5">
              <Download className="size-3.5" />
              <span className="font-bold">{formatNumber(model.downloads)}</span>
            </div>
            <div className="flex items-center gap-1.5 bg-black/40 backdrop-blur-sm rounded-full px-3 py-1.5">
              <Heart className="size-3.5" />
              <span className="font-bold">{formatNumber(model.likes)}</span>
            </div>
          </div>

          {/* Author info */}
          {model.author && (
            <div className="flex items-center gap-2 mb-2">
              <div className="flex items-center gap-1.5 text-white drop-shadow-lg">
                <User className="size-4" />
                <span className="text-sm font-medium truncate">{model.author}</span>
              </div>
            </div>
          )}

          {/* Model name */}
          <h3 className="text-xl font-bold text-white drop-shadow-lg line-clamp-2 leading-tight">{model.name}</h3>
        </div>
      </div>

      {/* Footer with size badge */}
      <CardFooter className="flex items-center justify-between border-t border-border/50 pt-3 pb-3 px-4 bg-muted/20">
        <Badge variant="outline" className="font-mono text-xs">
          {model.size}
        </Badge>

        {/* Additional tags if any */}
        {model.tags.length > 2 && (
          <Badge variant="outline" className="text-xs">
            +{model.tags.length - 2} more
          </Badge>
        )}
      </CardFooter>
    </Card>
  )

  // TEAM-479: Wrap in Link if href is provided
  if (href) {
    return (
      <Link href={href} className="block h-full">
        {cardContent}
      </Link>
    )
  }

  return cardContent
}
