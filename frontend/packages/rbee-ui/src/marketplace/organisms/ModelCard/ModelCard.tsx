// TEAM-405: Beautiful marketplace model card with modern design
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@rbee/ui/atoms/Card'
import { Download, Heart, User, Sparkles } from 'lucide-react'
import * as React from 'react'

export interface ModelCardProps {
  model: {
    id: string
    name: string
    description: string
    author?: string
    imageUrl?: string
    tags: string[]
    downloads: number
    likes: number
    size: string
  }
  onAction?: (modelId: string) => void
  actionButton?: React.ReactNode
  onClick?: () => void
}

export function ModelCard({ model, onAction, actionButton, onClick }: ModelCardProps) {
  // TEAM-422: Removed useState for SSG compatibility
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <Card 
      className="group h-full flex flex-col overflow-hidden border-border/50 hover:border-primary/50 transition-all duration-300 hover:shadow-xl hover:shadow-primary/5 hover:-translate-y-1"
    >
      {/* Image Section with Gradient Overlay */}
      {model.imageUrl ? (
        <div className="relative w-full aspect-video overflow-hidden bg-gradient-to-br from-primary/10 via-background to-muted">
          <img
            src={model.imageUrl}
            alt={model.name}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            loading="lazy"
          />
          {/* Gradient overlay for better text readability */}
          <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-background/20 to-transparent" />
          
          {/* Floating stats on image */}
          <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
            <div className="flex items-center gap-3 text-xs font-medium text-white drop-shadow-lg">
              <div className="flex items-center gap-1 bg-black/40 backdrop-blur-sm rounded-full px-2 py-1">
                <Download className="size-3" />
                <span>{formatNumber(model.downloads)}</span>
              </div>
              <div className="flex items-center gap-1 bg-black/40 backdrop-blur-sm rounded-full px-2 py-1">
                <Heart className="size-3" />
                <span>{formatNumber(model.likes)}</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Fallback gradient when no image
        <div className="relative w-full aspect-video bg-gradient-to-br from-primary/20 via-primary/10 to-muted flex items-center justify-center">
          <Sparkles className="size-12 text-primary/30" />
        </div>
      )}

      {/* Content Section */}
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg font-bold truncate group-hover:text-primary transition-colors">
              {model.name}
            </CardTitle>
            {model.author && (
              <div className="flex items-center gap-1.5 mt-1.5 text-xs text-muted-foreground">
                <User className="size-3.5" />
                <span className="truncate font-medium">{model.author}</span>
              </div>
            )}
          </div>
          {actionButton && <CardAction>{actionButton}</CardAction>}
        </div>
      </CardHeader>

      <CardContent className="flex-1 pt-0 pb-4">
        <CardDescription className="line-clamp-2 text-sm leading-relaxed mb-4">
          {model.description}
        </CardDescription>

        {/* Tags with better styling */}
        <div className="flex flex-wrap gap-1.5">
          {model.tags.slice(0, 4).map((tag) => (
            <Badge 
              key={tag} 
              variant="secondary"
              className="text-xs font-medium hover:bg-primary/20 transition-colors"
            >
              {tag}
            </Badge>
          ))}
          {model.tags.length > 4 && (
            <Badge variant="outline" className="text-xs">
              +{model.tags.length - 4}
            </Badge>
          )}
        </div>
      </CardContent>

      {/* Footer with action */}
      <CardFooter className="flex items-center justify-between border-t border-border/50 pt-4 bg-muted/20">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-xs">
            {model.size}
          </Badge>
        </div>

        {!actionButton && onAction && (
          <Button 
            size="sm" 
            onClick={() => onAction(model.id)}
            className="transition-all duration-200 hover:shadow-md"
          >
            <Download className="size-3.5 mr-1.5" />
            Download
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
