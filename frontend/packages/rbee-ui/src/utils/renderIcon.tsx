// Utility to render icons from either LucideIcon components or string names
// Supports SSG by allowing string names in config files

import type { LucideIcon, LucideProps } from 'lucide-react'
import * as LucideIcons from 'lucide-react'

export function renderIcon(icon: LucideIcon | string, props?: Omit<LucideProps, 'ref'>): React.ReactNode | null {
  // If it's a string, dynamically get the icon from lucide-react
  // Check string FIRST to avoid issues with SSR/SSG where typeof checks can be unreliable
  if (typeof icon === 'string') {
    const IconComponent = (LucideIcons as any)[icon] as LucideIcon | undefined
    if (!IconComponent) {
      console.warn(`Icon "${icon}" not found in lucide-react`)
      return null
    }
    return <IconComponent {...props} />
  }

  // If it's already a component, render it directly
  // This works in CSR but may be unreliable in SSR/SSG builds
  if (typeof icon === 'function') {
    const IconComponent = icon
    return <IconComponent {...props} />
  }

  return null
}
