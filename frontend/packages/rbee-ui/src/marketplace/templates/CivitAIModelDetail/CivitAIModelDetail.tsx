// TEAM-478: Enhanced CivitAI model detail to match CivitAI.com
// Added: versions, multiple images, full stats, all files, reviews

'use client'

import { Badge, Button, Card } from '@rbee/ui/atoms'
import { MarkdownContent } from '@rbee/ui/molecules'
import { ExternalLink } from 'lucide-react'
import { useState } from 'react'
import { CivitAIDetailsCard } from '../../organisms/CivitAIDetailsCard'
import { CivitAIFileCard } from '../../organisms/CivitAIFileCard'
import { CivitAIImageGallery } from '../../organisms/CivitAIImageGallery'
import { CivitAITrainedWords } from '../../organisms/CivitAITrainedWords'
import { ModelStats } from '../../organisms/ModelStats'

export interface CivitAIFile {
  name: string
  id: number
  sizeKb: number
  downloadUrl: string
  primary: boolean
}

export interface CivitAIImage {
  url: string
  nsfw: boolean
  width: number
  height: number
}

export interface CivitAIModelVersion {
  id: number
  name: string
  baseModel?: string
  description?: string
  createdAt: string
  updatedAt: string
  publishedAt?: string
  trainedWords?: string[]
  images: CivitAIImage[]
  files: CivitAIFile[]
  downloadUrl?: string
  stats?: {
    downloadCount: number
    ratingCount: number
    rating: number
    thumbsUpCount: number
  }
}

export interface CivitAIModelDetailProps {
  model: {
    id: string
    name: string
    description: string
    author: string
    downloads: number
    likes: number
    rating: number
    thumbsUpCount?: number
    commentCount?: number
    ratingCount?: number
    size: string
    tags: string[]
    type: string
    allowCommercialUse: string | boolean | unknown
    externalUrl?: string
    externalLabel?: string
    // TEAM-478: REQUIRED - versions array (no backwards compatibility)
    versions: CivitAIModelVersion[]
    publishedAt?: string
    updatedAt?: string
  }
}

export function CivitAIModelDetail({ model }: CivitAIModelDetailProps) {
  // TEAM-478: Version selector state
  const [selectedVersionIndex, setSelectedVersionIndex] = useState(0)

  // TEAM-478: RULE ZERO - No backwards compatibility, versions is required
  const currentVersion = model.versions[selectedVersionIndex]
  if (!currentVersion) {
    throw new Error('CivitAIModelDetail: versions array is required and must not be empty')
  }

  return (
    <div className="space-y-6">
      {/* Version Selector - TEAM-478: Show all versions like CivitAI.com */}
      {model.versions.length > 1 && (
        <div className="flex items-center gap-2 overflow-x-auto">
          {model.versions.map((version, index) => (
            <Button
              key={version.id}
              variant={index === selectedVersionIndex ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedVersionIndex(index)}
              className="whitespace-nowrap"
            >
              {version.name}
            </Button>
          ))}
        </div>
      )}

      {/* Main Content Grid - 3 columns: 2 for content, 1 for sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Columns (span 2) - Images & Description */}
        <div className="lg:col-span-2 space-y-6">
          {/* Image Gallery - TEAM-478: Show ALL images from selected version */}
          <CivitAIImageGallery images={currentVersion.images} modelName={model.name} />

          {/* Description - TEAM-478: Show main model description only */}
          {model.description && <MarkdownContent html={model.description} asCard={false} className="prose-sm" />}
        </div>

        {/* Right Column (span 1) - Details Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Title, Author, Stats, and External Link */}
          <div className="space-y-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight mb-2">{model.name}</h1>
              <p className="text-lg text-muted-foreground">
                by <span className="font-semibold">{model.author}</span>
              </p>
            </div>

            {/* TEAM-478: Enhanced Stats - Reusable ModelStats component */}
            <ModelStats
              downloads={currentVersion.stats?.downloadCount ?? model.downloads}
              likes={model.likes}
              rating={currentVersion.stats?.rating ?? model.rating}
              {...((currentVersion.stats?.thumbsUpCount ?? model.thumbsUpCount)
                ? { thumbsUpCount: currentVersion.stats?.thumbsUpCount ?? model.thumbsUpCount }
                : {})}
              {...(model.commentCount ? { commentCount: model.commentCount } : {})}
              {...(currentVersion.updatedAt || model.updatedAt || currentVersion.publishedAt || model.publishedAt
                ? {
                    updatedAt:
                      currentVersion.updatedAt || model.updatedAt || currentVersion.publishedAt || model.publishedAt,
                  }
                : {})}
            />

            {model.externalUrl && (
              <Button variant="outline" size="lg" className="w-full shadow-md hover:shadow-lg transition-all" asChild>
                <a href={model.externalUrl} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="size-4 mr-2" />
                  {model.externalLabel || 'View on CivitAI'}
                </a>
              </Button>
            )}
          </div>

          {/* Version Notes - TEAM-478: Moved to sidebar */}
          {currentVersion?.description && (
            <Card className="p-6 space-y-4 shadow-lg">
              <h3 className="font-semibold text-lg">Version Notes</h3>
              <MarkdownContent html={currentVersion.description} asCard={false} className="prose-sm" />
            </Card>
          )}

          {/* Details Card */}
          <CivitAIDetailsCard
            type={model.type}
            baseModel={currentVersion.baseModel || 'Unknown'}
            version={currentVersion.name}
            size={model.size}
            allowCommercialUse={model.allowCommercialUse}
          />

          {/* Trained Words */}
          <CivitAITrainedWords
            {...(currentVersion.trainedWords ? { trainedWords: currentVersion.trainedWords } : {})}
          />

          {/* Files - TEAM-478: Show ALL files from selected version */}
          <CivitAIFileCard files={currentVersion.files} />

          {/* Tags */}
          {model.tags.length > 0 && (
            <Card className="p-6 space-y-4 shadow-lg">
              <h3 className="font-semibold text-lg">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {model.tags.slice(0, 10).map((tag) => (
                  <Badge
                    key={tag}
                    variant="outline"
                    className="text-xs hover:bg-primary hover:text-primary-foreground transition-colors cursor-default"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
