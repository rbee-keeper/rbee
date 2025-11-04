// TEAM-405: Model metadata display card
//! Reusable component for displaying model metadata in a clean format

import { Card, CardContent, CardHeader, CardTitle, KeyValuePair } from '@rbee/ui/atoms'

export interface ModelMetadataItem {
  label: string
  value: string | number | React.ReactNode
  /** Optional icon */
  icon?: React.ReactNode
  /** Render as code block */
  code?: boolean
}

export interface ModelMetadataCardProps {
  title: string
  items: ModelMetadataItem[]
  className?: string
}

/**
 * Card for displaying model metadata in a clean key-value format
 * 
 * @example
 * ```tsx
 * <ModelMetadataCard
 *   title="Model Information"
 *   items={[
 *     { label: 'Architecture', value: 'LlamaForCausalLM' },
 *     { label: 'Model Type', value: 'llama' },
 *     { label: 'Context Length', value: '4096 tokens' }
 *   ]}
 * />
 * ```
 */
export function ModelMetadataCard({ title, items, className }: ModelMetadataCardProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {items.map((item, index) => (
          <div key={index} className="space-y-1">
            <div className="flex items-center gap-2">
              {item.icon}
              <p className="text-sm font-medium text-muted-foreground">{item.label}</p>
            </div>
            {item.code ? (
              <code className="block text-sm bg-muted px-3 py-2 rounded font-mono break-all">
                {item.value}
              </code>
            ) : (
              <p className="text-sm font-semibold">{item.value}</p>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
