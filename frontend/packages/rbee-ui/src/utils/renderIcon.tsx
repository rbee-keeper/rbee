// Utility to render icons from either LucideIcon components or string names
// Supports SSG by allowing string names in config files

import type { LucideIcon, LucideProps } from 'lucide-react'
import * as LucideIcons from 'lucide-react'

export function renderIcon(icon: LucideIcon | string, props?: Omit<LucideProps, 'ref'>): React.ReactNode | null {
  // If it's already a component, render it directly
  if (typeof icon === 'function') {
    const IconComponent = icon
    return <IconComponent {...props} />
  }

  // If it's a string, dynamically get the icon from lucide-react
  if (typeof icon === 'string') {
    const IconComponent = (LucideIcons as any)[icon] as LucideIcon | undefined
    if (!IconComponent) {
      console.warn(`Icon "${icon}" not found in lucide-react`)
      return null
    }
    return <IconComponent {...props} />
  }

  return null
}
