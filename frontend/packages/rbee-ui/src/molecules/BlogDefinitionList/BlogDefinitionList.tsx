import { cn } from '@rbee/ui/utils'
import type { VariantProps } from 'class-variance-authority'
import { cva } from 'class-variance-authority'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Variants
// ──────────────────────────────────────────────────────────────────────────────

const blogDefinitionListVariants = cva('space-y-4 my-6', {
  variants: {
    variant: {
      default: '',
      compact: 'space-y-2',
      relaxed: 'space-y-6',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

const blogDefinitionItemVariants = cva('', {
  variants: {
    variant: {
      default: '',
      card: 'p-4 rounded-lg border border-border bg-card',
      highlight: 'p-4 rounded-lg bg-muted/50',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export interface BlogDefinitionItem {
  /** Term/title (e.g., "NVIDIA CUDA") */
  term: React.ReactNode
  /** Definition/description */
  definition: React.ReactNode
}

export interface BlogDefinitionListProps extends VariantProps<typeof blogDefinitionListVariants> {
  /** Definition items */
  items: BlogDefinitionItem[]
  /** Item variant */
  itemVariant?: 'default' | 'card' | 'highlight'
  /** Additional CSS classes */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────────

/**
 * BlogDefinitionList - Styled definition list for term/definition pairs
 *
 * Perfect for technical documentation where you need to explain terms,
 * features, or specifications.
 *
 * @example
 * ```tsx
 * <BlogDefinitionList
 *   items={[
 *     {
 *       term: <strong>NVIDIA CUDA:</strong>,
 *       definition: 'Dedicated VRAM (fast, but limited size)',
 *     },
 *     {
 *       term: <strong>Apple Metal:</strong>,
 *       definition: 'Unified memory (large capacity, shared with CPU)',
 *     },
 *   ]}
 * />
 *
 * // With card styling
 * <BlogDefinitionList
 *   itemVariant="card"
 *   items={[...]}
 * />
 * ```
 */
export function BlogDefinitionList({ variant, itemVariant = 'default', items, className }: BlogDefinitionListProps) {
  return (
    <dl className={cn(blogDefinitionListVariants({ variant }), className)}>
      {items.map((item, index) => {
        const itemKey = typeof item.term === 'string' ? `${index}-${item.term.slice(0, 20)}` : `def-${index}`
        return (
          <div key={itemKey} className={cn(blogDefinitionItemVariants({ variant: itemVariant }))}>
            <dt className="font-semibold text-foreground mb-1">{item.term}</dt>
            <dd className="text-muted-foreground ml-0">{item.definition}</dd>
          </div>
        )
      })}
    </dl>
  )
}

export { blogDefinitionListVariants, blogDefinitionItemVariants }
