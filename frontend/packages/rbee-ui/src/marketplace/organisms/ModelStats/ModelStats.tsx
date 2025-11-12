// TEAM-478: Reusable model stats component for CivitAI and HuggingFace
// Displays downloads, likes, ratings, comments, and update date

'use client'

import { Calendar, Download, Heart, MessageSquare, Star, ThumbsUp } from 'lucide-react'

export interface ModelStatsProps {
  downloads?: number
  likes?: number
  rating?: number
  thumbsUpCount?: number
  commentCount?: number
  updatedAt?: string
  className?: string
}

export function ModelStats({
  downloads,
  likes,
  rating,
  thumbsUpCount,
  commentCount,
  updatedAt,
  className,
}: ModelStatsProps) {
  return (
    <div className={className}>
      <div className="flex flex-wrap gap-4 text-sm">
        {/* Downloads */}
        {downloads !== undefined && downloads > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Download className="size-4" />
            <span>{downloads.toLocaleString()}</span>
          </div>
        )}

        {/* Likes */}
        {likes !== undefined && likes > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Heart className="size-4" />
            <span>{likes.toLocaleString()}</span>
          </div>
        )}

        {/* Rating */}
        {rating !== undefined && rating > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Star className="size-4 fill-yellow-400 text-yellow-400" />
            <span>{rating.toFixed(1)}</span>
          </div>
        )}

        {/* Thumbs Up */}
        {thumbsUpCount !== undefined && thumbsUpCount > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <ThumbsUp className="size-4" />
            <span>{thumbsUpCount.toLocaleString()}</span>
          </div>
        )}

        {/* Comments */}
        {commentCount !== undefined && commentCount > 0 && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <MessageSquare className="size-4" />
            <span>{commentCount.toLocaleString()}</span>
          </div>
        )}

        {/* Updated Date */}
        {updatedAt && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Calendar className="size-4" />
            <span className="text-xs">Updated {new Date(updatedAt).toLocaleDateString()}</span>
          </div>
        )}
      </div>
    </div>
  )
}
