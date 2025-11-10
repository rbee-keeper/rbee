import { Alert, AlertDescription, AlertTitle } from '@rbee/ui/atoms/Alert'
import { cn } from '@rbee/ui/utils'
import { AlertCircle, CheckCircle, Info, Lightbulb, AlertTriangle, Rocket, DollarSign } from 'lucide-react'
import type * as React from 'react'

export interface BlogCalloutProps {
  /** Callout type/variant */
  variant?: 'info' | 'success' | 'warning' | 'danger' | 'tip' | 'example' | 'pricing'
  /** Callout title */
  title: string
  /** Callout content */
  children: React.ReactNode
  /** Additional CSS classes */
  className?: string
}

const variantConfig = {
  info: {
    variant: 'info' as const,
    icon: Info,
    emoji: '💡',
  },
  success: {
    variant: 'success' as const,
    icon: CheckCircle,
    emoji: '✅',
  },
  warning: {
    variant: 'warning' as const,
    icon: AlertTriangle,
    emoji: '⚠️',
  },
  danger: {
    variant: 'destructive' as const,
    icon: AlertCircle,
    emoji: '❌',
  },
  tip: {
    variant: 'primary' as const,
    icon: Lightbulb,
    emoji: '💡',
  },
  example: {
    variant: 'default' as const,
    icon: Rocket,
    emoji: '🚀',
  },
  pricing: {
    variant: 'info' as const,
    icon: DollarSign,
    emoji: '💎',
  },
}

/**
 * BlogCallout - Enhanced callout boxes for blog posts
 * Provides consistent styling for tips, warnings, examples, and more
 * 
 * @example
 * <BlogCallout variant="tip" title="Pro Tip">
 *   Use worker metadata to tag workers with custom properties.
 * </BlogCallout>
 */
export function BlogCallout({ variant = 'info', title, children, className }: BlogCalloutProps) {
  const config = variantConfig[variant]
  const Icon = config.icon

  return (
    <Alert variant={config.variant} className={cn('my-6', className)}>
      <Icon className="h-4 w-4" />
      <AlertTitle>{config.emoji} {title}</AlertTitle>
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}
