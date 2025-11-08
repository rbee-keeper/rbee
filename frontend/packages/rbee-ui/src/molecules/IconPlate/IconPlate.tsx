import { cn, renderIcon, type IconName } from '@rbee/ui/utils'
import type * as React from 'react'

export interface IconPlateProps {
  /** Icon name string or rendered icon component */
  icon: IconName | React.ReactNode
  /** Size variant */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Color tone - supports both tone names and chart colors */
  tone?: 'primary' | 'muted' | 'destructive' | 'success' | 'warning' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
  /** Shape variant */
  shape?: 'square' | 'circle' | 'rounded'
  /** Additional CSS classes */
  className?: string
}

/**
 * IconPlate molecule - reusable icon container
 * Consolidates IconBox and 15+ instances of icon wrapper patterns
 * Used across features, stats, cards, and list items
 *
 * Accepts rendered icon components for flexibility.
 */
export function IconPlate({ icon, size = 'md', tone = 'primary', shape = 'square', className }: IconPlateProps) {
  const sizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12',
    xl: 'h-16 w-16',
  }

  const iconSizeClasses = {
    sm: 'size-4',
    md: 'size-6',
    lg: 'size-7',
    xl: 'size-10',
  }

  const toneClasses = {
    primary: 'bg-primary/10 text-primary',
    muted: 'bg-muted text-muted-foreground',
    destructive: 'bg-destructive/10 text-destructive',
    success: 'bg-chart-3/10 text-chart-3',
    warning: 'bg-chart-4/10 text-chart-4',
    'chart-1': 'bg-chart-1/10 text-chart-1',
    'chart-2': 'bg-chart-2/10 text-chart-2',
    'chart-3': 'bg-chart-3/10 text-chart-3',
    'chart-4': 'bg-chart-4/10 text-chart-4',
    'chart-5': 'bg-chart-5/10 text-chart-5',
  }

  const shapeClasses = {
    square: 'rounded',
    circle: 'rounded-full',
    rounded: 'rounded-lg',
  }

  return (
    <div
      className={cn(
        'flex shrink-0 items-center justify-center transition-colors',
        sizeClasses[size],
        toneClasses[tone],
        shapeClasses[shape],
        className,
      )}
      aria-hidden="true"
    >
      {typeof icon === 'string' ? renderIcon(icon as IconName, iconSizeClasses[size]) : icon}
    </div>
  )
}
