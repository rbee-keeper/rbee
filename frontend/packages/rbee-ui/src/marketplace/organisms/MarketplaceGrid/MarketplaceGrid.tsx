// TEAM-464: Simple grid layout for marketplace items
// Created to fix missing MarketplaceGrid import

import type { ReactNode } from 'react'

export interface MarketplaceGridProps {
  children: ReactNode
  className?: string
}

/**
 * Simple responsive grid for marketplace items
 */
export function MarketplaceGrid({ children, className = '' }: MarketplaceGridProps) {
  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 ${className}`}>
      {children}
    </div>
  )
}
