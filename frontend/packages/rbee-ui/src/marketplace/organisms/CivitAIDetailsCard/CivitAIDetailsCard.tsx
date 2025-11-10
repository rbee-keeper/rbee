// TEAM-463: Premium details card for CivitAI models
// Beautiful metadata display with icons and badges

'use client'

import { Badge, Card, Separator } from '@rbee/ui/atoms'
import { Package, Layers, Tag, HardDrive, ShieldCheck } from 'lucide-react'
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
  const commercialUseValue = String(allowCommercialUse || 'Unknown')
  const isAllowed = commercialUseValue.toLowerCase().includes('allowed') || 
                    commercialUseValue.toLowerCase().includes('yes') ||
                    commercialUseValue === 'true'

  const details = [
    {
      icon: Package,
      label: 'Type',
      value: type,
      badge: true,
      color: 'text-purple-500',
    },
    {
      icon: Layers,
      label: 'Base Model',
      value: baseModel,
      badge: false,
      color: 'text-blue-500',
    },
    {
      icon: Tag,
      label: 'Version',
      value: version,
      badge: false,
      color: 'text-green-500',
    },
    {
      icon: HardDrive,
      label: 'Size',
      value: size,
      badge: false,
      color: 'text-orange-500',
    },
    {
      icon: ShieldCheck,
      label: 'Commercial Use',
      value: commercialUseValue,
      badge: true,
      color: isAllowed ? 'text-green-500' : 'text-red-500',
    },
  ]

  return (
    <Card className={cn('p-6 space-y-4', className)}>
      <h3 className="font-semibold text-lg flex items-center gap-2">
        <Package className="size-5 text-primary" />
        Details
      </h3>

      <div className="space-y-3">
        {details.map((detail, idx) => {
          const Icon = detail.icon
          return (
            <div key={detail.label}>
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Icon className={cn('size-4', detail.color)} />
                  <span>{detail.label}</span>
                </div>
                {detail.badge ? (
                  <Badge
                    variant={detail.label === 'Commercial Use' && detail.value.toLowerCase().includes('allowed') ? 'default' : 'secondary'}
                    className="font-medium"
                  >
                    {detail.value}
                  </Badge>
                ) : (
                  <span className="font-medium text-sm">{detail.value}</span>
                )}
              </div>
              {idx < details.length - 1 && <Separator />}
            </div>
          )
        })}
      </div>
    </Card>
  )
}
