import { cn, renderIcon, type IconName } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface ComplianceChipProps {
  /** Icon name string or icon element */
  icon?: IconName | ReactNode
  /** Chip label */
  children: ReactNode
  /** Accessible label for screen readers */
  ariaLabel?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * ComplianceChip molecule for compliance proof indicators
 * Compact, chip-style badges with optional icons
 */
export function ComplianceChip({ icon, children, ariaLabel, className }: ComplianceChipProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 rounded border border-border/60 bg-background/50 px-2 py-1 text-xs font-medium text-muted-foreground backdrop-blur-sm transition-colors hover:border-border hover:bg-background/80',
        className,
      )}
      aria-label={ariaLabel}
    >
      {icon && (
        <div className="shrink-0">
          {typeof icon === 'string' ? renderIcon(icon as IconName, 'size-3') : icon}
        </div>
      )}
      <span>{children}</span>
    </div>
  )
}
