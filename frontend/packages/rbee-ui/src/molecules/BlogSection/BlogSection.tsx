import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export interface BlogSectionProps {
  /** Section content */
  children: React.ReactNode
  /** Additional CSS classes */
  className?: string
  /** Section ID for anchor links */
  id?: string
  /** Remove prose styling (for custom components) */
  noProse?: boolean
}

// ──────────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────────

/**
 * BlogSection - Wrapper for blog content sections
 * 
 * Provides consistent spacing and prose styling for blog content.
 * Use `noProse` prop to wrap custom components that don't need prose styles.
 * 
 * @example
 * ```tsx
 * <BlogSection>
 *   <BlogHeading level="h2">Introduction</BlogHeading>
 *   <p>Some content...</p>
 * </BlogSection>
 * 
 * <BlogSection noProse>
 *   <StatsGrid stats={[...]} />
 * </BlogSection>
 * ```
 */
export function BlogSection({ children, className, id, noProse = false }: BlogSectionProps) {
  return (
    <section id={id} className={cn('my-8', !noProse && 'prose-section', className)}>
      {children}
    </section>
  )
}
