// TEAM-463: Premium details card for CivitAI models
// Clean, professional metadata display
// TEAM-481: Redesigned with minimal, professional styling - removed colorful icons

'use client'

import { Badge, Card } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'

export interface CivitAIDetailsCardProps {
  type: string
  baseModel: string
  version: string
  size: string
  allowCommercialUse: string | boolean | unknown // TEAM-463: CivitAI API returns various types
  className?: string
}

export function CivitAIDetailsCard({
  type,
  baseModel,
  version,
  size,
  allowCommercialUse,
  className,
}: CivitAIDetailsCardProps) {
  // TEAM-463: Safely handle allowCommercialUse which might be string, boolean, or object
  // TEAM-481: Enhanced detection for various CivitAI commercial use formats
  const commercialUseValue = String(allowCommercialUse || 'Unknown')

  // Check if commercial use is allowed
  // CivitAI returns values like: "Image", "RentCivit", "Sell", "None", etc.
  const isAllowed =
    commercialUseValue.toLowerCase().includes('image') ||
    commercialUseValue.toLowerCase().includes('rent') ||
    commercialUseValue.toLowerCase().includes('sell') ||
    commercialUseValue.toLowerCase().includes('allowed') ||
    commercialUseValue.toLowerCase().includes('yes') ||
    commercialUseValue === 'true'

  const isRestricted =
    commercialUseValue.toLowerCase().includes('none') ||
    commercialUseValue.toLowerCase() === 'no' ||
    commercialUseValue === 'false'

  const details = [
    {
      label: 'Type',
      value: type,
      badge: true,
    },
    {
      label: 'Base Model',
      value: baseModel,
      badge: false,
    },
    {
      label: 'Version',
      value: version,
      badge: false,
    },
    {
      label: 'Size',
      value: size,
      badge: false,
    },
    {
      label: 'Commercial Use',
      value: commercialUseValue,
      badge: true,
      isCommercial: true,
    },
  ]

  return (
    <Card className={cn('p-6 border-border/50', className)}>
      <div className="space-y-1 mb-6">
        <h3 className="font-semibold text-base tracking-tight">Details</h3>
        <p className="text-xs text-muted-foreground">Model specifications and licensing</p>
      </div>

      <div className="space-y-4">
        {details.map((detail) => {
          return (
            <div key={detail.label} className="flex items-start justify-between gap-4">
              <span className="text-sm text-muted-foreground font-medium min-w-[120px]">{detail.label}</span>
              {detail.badge ? (
                <Badge
                  variant="outline"
                  className={cn(
                    'font-medium text-xs',
                    detail.isCommercial &&
                      isAllowed &&
                      'bg-green-500/10 text-green-600 hover:bg-green-500/15 border-green-500/30',
                    detail.isCommercial &&
                      isRestricted &&
                      'bg-red-500/10 text-red-600 hover:bg-red-500/15 border-red-500/30',
                    detail.isCommercial &&
                      !isAllowed &&
                      !isRestricted &&
                      'bg-yellow-500/10 text-yellow-600 hover:bg-yellow-500/15 border-yellow-500/30',
                    !detail.isCommercial && 'bg-muted/50',
                  )}
                >
                  {detail.value}
                </Badge>
              ) : (
                <span className="font-semibold text-sm text-right flex-1">{detail.value}</span>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}
