import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utility function to merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// TEAM-421: Re-export environment detection utilities
export * from './environment'
// Re-export focus ring utilities
export { brandLink, focusRing, focusRingDestructive, focusRingTight } from './focus-ring'
// Re-export utilities
export * from './iconMap'
export * from './parse-inline-markdown'
export { renderIcon } from './renderIcon'
