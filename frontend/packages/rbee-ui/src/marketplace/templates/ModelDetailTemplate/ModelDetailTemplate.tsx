// TEAM-401: DUMB template
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms/Card'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Calendar, Download, ExternalLink, Heart, Scale, User } from 'lucide-react'
import { MarketplaceGrid } from '../../organisms/MarketplaceGrid'
import { ModelCard } from '../../organisms/ModelCard'
import type { ModelDetailTemplateProps } from './ModelDetailTemplateProps'

export function ModelDetailTemplate({
  model,
  installButton,
  relatedModels = [],
  onRelatedModelAction,
}: ModelDetailTemplateProps) {
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const formatDate = (dateString?: string): string => {
    if (!dateString) return 'Unknown'
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Image */}
        <div className="lg:col-span-1">
          {model.imageUrl ? (
            <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-muted">
              <img
                src={model.imageUrl}
                alt={model.name}
                className="w-full h-full object-cover"
              />
            </div>
          ) : (
            <div className="relative w-full aspect-square rounded-lg bg-muted flex items-center justify-center">
              <span className="text-4xl font-bold text-muted-foreground">
                {model.name.charAt(0)}
              </span>
            </div>
          )}
        </div>

        {/* Right: Info */}
        <div className="lg:col-span-2 space-y-4">
          <div>
            <h1 className="text-4xl font-serif font-bold tracking-tight mb-2">{model.name}</h1>
            {model.author && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <User className="size-4" />
                {model.authorUrl ? (
                  <a
                    href={model.authorUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-foreground underline underline-offset-4 flex items-center gap-1"
                  >
                    {model.author}
                    <ExternalLink className="size-3" />
                  </a>
                ) : (
                  <span>{model.author}</span>
                )}
              </div>
            )}
          </div>

          <p className="text-lg text-muted-foreground">{model.description}</p>

          {/* Tags */}
          <div className="flex flex-wrap gap-2">
            {model.tags.map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>

          {/* Stats */}
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <Download className="size-4 text-muted-foreground" />
              <span className="font-medium">{formatNumber(model.downloads)}</span>
              <span className="text-muted-foreground">downloads</span>
            </div>
            <div className="flex items-center gap-2">
              <Heart className="size-4 text-muted-foreground" />
              <span className="font-medium">{formatNumber(model.likes)}</span>
              <span className="text-muted-foreground">likes</span>
            </div>
          </div>

          {/* Install Button */}
          {installButton || (
            <Button size="lg" className="w-full sm:w-auto">
              <Download className="size-4" />
              Download Model ({model.size})
            </Button>
          )}
        </div>
      </div>

      <Separator />

      {/* Details Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {model.longDescription && (
            <Card>
              <CardHeader>
                <CardTitle>About</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <p className="whitespace-pre-wrap">{model.longDescription}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar: Specs */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Specifications</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {model.parameters && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Parameters</span>
                  <span className="text-sm font-medium">{model.parameters}</span>
                </div>
              )}
              {model.quantization && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Quantization</span>
                  <span className="text-sm font-medium">{model.quantization}</span>
                </div>
              )}
              {model.contextLength && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Context Length</span>
                  <span className="text-sm font-medium">{model.contextLength}</span>
                </div>
              )}
              {model.architecture && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Architecture</span>
                  <span className="text-sm font-medium">{model.architecture}</span>
                </div>
              )}
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Size</span>
                <span className="text-sm font-medium">{model.size}</span>
              </div>
              {model.license && (
                <div className="flex items-start justify-between gap-2">
                  <span className="text-sm text-muted-foreground flex items-center gap-1">
                    <Scale className="size-3" />
                    License
                  </span>
                  <span className="text-sm font-medium text-right">{model.license}</span>
                </div>
              )}
              {model.createdAt && (
                <div className="flex items-start justify-between gap-2">
                  <span className="text-sm text-muted-foreground flex items-center gap-1">
                    <Calendar className="size-3" />
                    Created
                  </span>
                  <span className="text-sm font-medium text-right">{formatDate(model.createdAt)}</span>
                </div>
              )}
              {model.updatedAt && (
                <div className="flex items-start justify-between gap-2">
                  <span className="text-sm text-muted-foreground flex items-center gap-1">
                    <Calendar className="size-3" />
                    Updated
                  </span>
                  <span className="text-sm font-medium text-right">{formatDate(model.updatedAt)}</span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Related Models */}
      {relatedModels.length > 0 && (
        <>
          <Separator />
          <div className="space-y-4">
            <h2 className="text-2xl font-serif font-bold tracking-tight">Related Models</h2>
            <MarketplaceGrid
              items={relatedModels}
              renderItem={(relatedModel) => (
                <ModelCard
                  key={relatedModel.id}
                  model={relatedModel}
                  onAction={onRelatedModelAction}
                />
              )}
              columns={3}
            />
          </div>
        </>
      )}
    </div>
  )
}
