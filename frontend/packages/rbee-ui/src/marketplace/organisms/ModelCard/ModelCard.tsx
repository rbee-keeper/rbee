// TEAM-401: Marketplace model card organism
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
import { Download, Heart, User } from 'lucide-react'
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
}

export function ModelCard({ model, onAction, actionButton }: ModelCardProps) {
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <Card className="h-full flex flex-col hover:shadow-md transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <CardTitle className="truncate">{model.name}</CardTitle>
            {model.author && (
              <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                <User className="size-3" />
                <span className="truncate">{model.author}</span>
              </div>
            )}
          </div>
          {actionButton && <CardAction>{actionButton}</CardAction>}
        </div>
        <CardDescription className="line-clamp-2">{model.description}</CardDescription>
      </CardHeader>

      <CardContent className="flex-1">
        {model.imageUrl && (
          <div className="relative w-full aspect-video rounded-md overflow-hidden bg-muted mb-3">
            <img
              src={model.imageUrl}
              alt={model.name}
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        )}

        <div className="flex flex-wrap gap-1.5">
          {model.tags.slice(0, 3).map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
          {model.tags.length > 3 && (
            <Badge variant="outline">+{model.tags.length - 3}</Badge>
          )}
        </div>
      </CardContent>

      <CardFooter className="flex items-center justify-between border-t pt-4">
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Download className="size-3" />
            <span>{formatNumber(model.downloads)}</span>
          </div>
          <div className="flex items-center gap-1">
            <Heart className="size-3" />
            <span>{formatNumber(model.likes)}</span>
          </div>
          <span className="font-medium">{model.size}</span>
        </div>

        {!actionButton && onAction && (
          <Button size="sm" onClick={() => onAction(model.id)}>
            Download
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
