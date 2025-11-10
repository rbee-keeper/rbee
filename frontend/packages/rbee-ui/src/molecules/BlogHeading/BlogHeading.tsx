import { cn } from '@rbee/ui/utils'
import { cva } from 'class-variance-authority'
import type { VariantProps } from 'class-variance-authority'
import type * as React from 'react'
import type { ElementType } from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Variants
// ──────────────────────────────────────────────────────────────────────────────

const blogHeadingVariants = cva('font-bold tracking-tight scroll-mt-20', {
  variants: {
    level: {
      h2: 'text-3xl md:text-4xl mt-12 mb-6 pb-2 border-b border-border',
      h3: 'text-2xl md:text-3xl mt-10 mb-4',
      h4: 'text-xl md:text-2xl mt-8 mb-3',
      h5: 'text-lg md:text-xl mt-6 mb-2',
      h6: 'text-base md:text-lg mt-4 mb-2',
    },
    variant: {
      default: '',
      gradient: 'bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent',
      accent: 'text-primary',
    },
  },
  defaultVariants: {
    level: 'h2',
    variant: 'default',
  },
})

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export interface BlogHeadingProps extends VariantProps<typeof blogHeadingVariants> {
  /** Heading text */
  children: React.ReactNode
  /** Optional ID for anchor links */
  id?: string
  /** Additional CSS classes */
  className?: string
  /** Show anchor link icon on hover */
  showAnchor?: boolean
}

// ──────────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────────

export function BlogHeading({ level = 'h2', variant, children, id, className, showAnchor = true }: BlogHeadingProps) {
  const Component = level as ElementType
  const headingId = id || (typeof children === 'string' ? children.toLowerCase().replace(/\s+/g, '-') : undefined)

  return (
    <Component id={headingId} className={cn(blogHeadingVariants({ level, variant }), 'group', className)}>
      {children}
      {showAnchor && headingId && (
        <a
          href={`#${headingId}`}
          className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-primary no-underline"
          aria-label={`Link to ${children}`}
        >
          #
        </a>
      )}
    </Component>
  )
}

export { blogHeadingVariants }
