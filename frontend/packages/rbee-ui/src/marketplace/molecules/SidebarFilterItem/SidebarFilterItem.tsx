// TEAM_511: Sidebar filter selection card for marketplace sidebars
import { Slot } from '@radix-ui/react-slot'
import { cn } from '@rbee/ui/utils'
import { cva, type VariantProps } from 'class-variance-authority'
import type * as React from 'react'

const sidebarFilterItemVariants = cva(
  'flex items-start gap-3 rounded-lg border transition-all',
  {
    variants: {
      state: {
        default: 'bg-muted border-sidebar-border hover:bg-muted/80',
        selected: 'bg-sidebar-accent/10 border-sidebar-accent hover:bg-sidebar-accent/20',
      },
      size: {
        sm: 'p-2',
        md: 'p-3',
      },
    },
    defaultVariants: {
      state: 'default',
      size: 'sm',
    },
  },
)

export interface SidebarFilterItemProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof sidebarFilterItemVariants> {
  asChild?: boolean
  selected?: boolean
}

export function SidebarFilterItem({
  asChild,
  selected,
  size,
  className,
  ...props
}: SidebarFilterItemProps) {
  const Comp = asChild ? Slot : 'div'
  const state = selected ? 'selected' : 'default'

  return (
    <Comp
      data-slot="sidebar-filter-item"
      className={cn(sidebarFilterItemVariants({ state, size, className }))}
      {...props}
    />
  )
}

export { sidebarFilterItemVariants }
