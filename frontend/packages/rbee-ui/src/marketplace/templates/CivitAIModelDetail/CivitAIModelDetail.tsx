// TEAM-463: Premium CivitAI-style model detail page
// Beautiful custom components with Next.js Image, animations, and professional styling

'use client'

import { Badge, Button, Card } from '@rbee/ui/atoms'
import { ExternalLink } from 'lucide-react'
import { MarkdownContent } from '@rbee/ui/molecules'
import { CivitAIImageGallery } from '../../organisms/CivitAIImageGallery'
import { CivitAIStatsHeader } from '../../organisms/CivitAIStatsHeader'
import { CivitAIDetailsCard } from '../../organisms/CivitAIDetailsCard'
import { CivitAIFileCard } from '../../organisms/CivitAIFileCard'
import { CivitAITrainedWords } from '../../organisms/CivitAITrainedWords'

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

export interface CivitAIModelDetailProps {
  model: {
    id: string
    name: string
    description: string
    author: string
    downloads: number
    likes: number
    rating: number
    size: string
    tags: string[]
    type: string
    baseModel: string
    version: string
    images: CivitAIImage[]
    files: CivitAIFile[]
    trainedWords?: string[]
    allowCommercialUse: string | boolean | unknown // TEAM-463: CivitAI API returns various types
    externalUrl?: string
    externalLabel?: string
  }
}

export function CivitAIModelDetail({ model }: CivitAIModelDetailProps) {
  return (
    <div className="space-y-6">
      {/* Stats Header */}
      <CivitAIStatsHeader
        downloads={model.downloads}
        likes={model.likes}
        rating={model.rating}
      />

      {/* Main Content Grid - 3 columns: 2 for content, 1 for sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Columns (span 2) - Images & Description */}
        <div className="lg:col-span-2 space-y-6">
          {/* Image Gallery */}
          <CivitAIImageGallery images={model.images} modelName={model.name} />

          {/* Description */}
          <Card className="p-6 shadow-lg">
            {/* Description is HTML from CivitAI API - converted to markdown for styling */}
            <MarkdownContent html={model.description} asCard={false} className="prose-sm" />
          </Card>
        </div>

        {/* Right Column (span 1) - Details Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Details Card */}
          <CivitAIDetailsCard
            type={model.type}
            baseModel={model.baseModel}
            version={model.version}
            size={model.size}
            allowCommercialUse={model.allowCommercialUse}
          />

          {/* Trained Words */}
          <CivitAITrainedWords trainedWords={model.trainedWords} />

          {/* Files */}
          <CivitAIFileCard files={model.files} />

          {/* Tags */}
          {model.tags.length > 0 && (
            <Card className="p-6 space-y-4 shadow-lg">
              <h3 className="font-semibold text-lg">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {model.tags.slice(0, 10).map((tag) => (
                  <Badge key={tag} variant="outline" className="text-xs hover:bg-primary hover:text-primary-foreground transition-colors cursor-default">
                    {tag}
                  </Badge>
                ))}
              </div>
            </Card>
          )}

          {/* External Link */}
          {model.externalUrl && (
            <Button
              variant="outline"
              size="lg"
              className="w-full shadow-md hover:shadow-lg transition-all"
              asChild
            >
              <a href={model.externalUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="size-4 mr-2" />
                {model.externalLabel || 'View Original'}
              </a>
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
