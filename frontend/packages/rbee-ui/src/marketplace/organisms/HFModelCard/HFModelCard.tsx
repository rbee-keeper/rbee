// TEAM-464: HuggingFace Model Card component
// Displays structured metadata from model card YAML frontmatter
// Separate from README markdown content

'use client'

import { Badge, Card, Separator } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import { BookOpen, Code, Database, Globe, Shield } from 'lucide-react'

export interface HFModelCardProps {
  cardData?: {
    base_model?: string
    license?: string
    language?: string | string[]
    datasets?: string[]
    pipeline_tag?: string
  }
  pipelineTag?: string
  libraryName?: string
  className?: string
}

export function HFModelCard({ cardData, pipelineTag, libraryName, className }: HFModelCardProps) {
  // Don't render if no data
  if (!cardData && !pipelineTag && !libraryName) {
    return null
  }

  const items = [
    // Pipeline Tag
    ...(pipelineTag || cardData?.pipeline_tag
      ? [
          {
            icon: Code,
            label: 'Pipeline',
            value: pipelineTag || cardData?.pipeline_tag,
            badge: true,
            color: 'text-purple-500',
          },
        ]
      : []),
    // Library
    ...(libraryName
      ? [
          {
            icon: BookOpen,
            label: 'Library',
            value: libraryName,
            badge: true,
            color: 'text-blue-500',
          },
        ]
      : []),
    // Base Model
    ...(cardData?.base_model
      ? [
          {
            icon: Code,
            label: 'Base Model',
            value: cardData.base_model,
            badge: false,
            color: 'text-green-500',
          },
        ]
      : []),
    // License
    ...(cardData?.license
      ? [
          {
            icon: Shield,
            label: 'License',
            value: cardData.license,
            badge: true,
            color: 'text-orange-500',
          },
        ]
      : []),
    // Language
    ...(cardData?.language
      ? [
          {
            icon: Globe,
            label: 'Language',
            value:
              typeof cardData.language === 'string'
                ? cardData.language
                : cardData.language.slice(0, 3).join(', ') + (cardData.language.length > 3 ? '...' : ''),
            badge: false,
            color: 'text-cyan-500',
          },
        ]
      : []),
    // Datasets
    ...(cardData?.datasets && cardData.datasets.length > 0
      ? [
          {
            icon: Database,
            label: 'Datasets',
            value: `${cardData.datasets.length} dataset${cardData.datasets.length === 1 ? '' : 's'}`,
            badge: false,
            color: 'text-pink-500',
          },
        ]
      : []),
  ]

  // Don't render if no items
  if (items.length === 0) {
    return null
  }

  return (
    <Card className={cn('p-6 space-y-4 shadow-lg', className)}>
      <h3 className="font-semibold text-lg flex items-center gap-2">
        <BookOpen className="size-5 text-primary" />
        Model Card
      </h3>

      <div className="space-y-3">
        {items.map((item, idx) => {
          const Icon = item.icon
          return (
            <div key={item.label}>
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Icon className={cn('size-4', item.color)} />
                  <span>{item.label}</span>
                </div>
                {item.badge ? (
                  <Badge variant="secondary" className="font-medium">
                    {item.value}
                  </Badge>
                ) : (
                  <span className="font-medium text-sm">{item.value}</span>
                )}
              </div>
              {idx < items.length - 1 && <Separator />}
            </div>
          )
        })}
      </div>

      {/* Datasets List (if available) */}
      {cardData?.datasets && cardData.datasets.length > 0 && (
        <div className="pt-3 border-t">
          <p className="text-xs text-muted-foreground mb-2">Training Datasets:</p>
          <div className="flex flex-wrap gap-2">
            {cardData.datasets.slice(0, 5).map((dataset) => (
              <Badge key={dataset} variant="outline" className="text-xs">
                {dataset}
              </Badge>
            ))}
            {cardData.datasets.length > 5 && (
              <Badge variant="outline" className="text-xs">
                +{cardData.datasets.length - 5} more
              </Badge>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}
