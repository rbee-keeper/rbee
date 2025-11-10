import { cn } from '@rbee/ui/utils'
import type { VariantProps } from 'class-variance-authority'
import { cva } from 'class-variance-authority'
import { Check, ChevronRight, Circle, X } from 'lucide-react'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Variants
// ──────────────────────────────────────────────────────────────────────────────

const blogListVariants = cva('space-y-3 my-6', {
  variants: {
    variant: {
      default: 'list-disc list-inside',
      ordered: 'list-decimal list-inside',
      checklist: 'list-none',
      pros: 'list-none',
      cons: 'list-none',
      steps: 'list-none',
    },
    spacing: {
      compact: 'space-y-2',
      default: 'space-y-3',
      relaxed: 'space-y-4',
    },
  },
  defaultVariants: {
    variant: 'default',
    spacing: 'default',
  },
})

const blogListItemVariants = cva('flex items-start gap-2 text-base leading-relaxed', {
  variants: {
    variant: {
      default: '',
      ordered: '',
      checklist: '',
      pros: 'text-green-700 dark:text-green-400',
      cons: 'text-red-700 dark:text-red-400',
      steps: '',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export interface BlogListProps extends VariantProps<typeof blogListVariants> {
  /** List items */
  items: React.ReactNode[]
  /** Additional CSS classes */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────────

/**
 * BlogList - Styled list component for blog posts
 *
 * @example
 * ```tsx
 * <BlogList
 *   variant="checklist"
 *   items={[
 *     'Enable audit logging',
 *     'Set data retention policies',
 *     'Implement access controls',
 *   ]}
 * />
 *
 * <BlogList
 *   variant="pros"
 *   items={[
 *     'Full control over data',
 *     'No vendor lock-in',
 *     'Predictable costs',
 *   ]}
 * />
 * ```
 */
export function BlogList({ variant = 'default', spacing, items, className }: BlogListProps) {
  const ListComponent = variant === 'ordered' ? 'ol' : 'ul'

  const getIcon = (itemVariant: typeof variant) => {
    switch (itemVariant) {
      case 'checklist':
        return <Check className="h-5 w-5 text-primary shrink-0 mt-0.5" />
      case 'pros':
        return <Check className="h-5 w-5 text-green-600 dark:text-green-400 shrink-0 mt-0.5" />
      case 'cons':
        return <X className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0 mt-0.5" />
      case 'steps':
        return <ChevronRight className="h-5 w-5 text-primary shrink-0 mt-0.5" />
      default:
        return variant === 'default' ? <Circle className="h-2 w-2 fill-current shrink-0 mt-2" /> : null
    }
  }

  const showIcon = variant !== 'default' && variant !== 'ordered'

  return (
    <ListComponent className={cn(blogListVariants({ variant, spacing }), className)}>
      {items.map((item, index) => {
        // Use a combination of index and item content hash for key
        // This is acceptable for static list items that don't reorder
        const itemKey = typeof item === 'string' ? `${index}-${item.slice(0, 20)}` : `item-${index}`
        return (
          <li key={itemKey} className={cn(blogListItemVariants({ variant }), showIcon && 'list-none')}>
            {showIcon && getIcon(variant)}
            <span className="flex-1">{item}</span>
          </li>
        )
      })}
    </ListComponent>
  )
}

export { blogListVariants, blogListItemVariants }
