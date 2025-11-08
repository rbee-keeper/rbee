// TEAM-460: Shared types for Navigation organism and its molecules

import type { LucideIcon } from 'lucide-react'

export interface NavigationLink {
  label: string
  href: string
  description?: string
  icon?: LucideIcon | string // Support both component and string name for SSG
}

export interface NavigationCTA {
  label: string
  href?: string
  onClick?: () => void
}

export interface DropdownSection {
  type: 'dropdown'
  title: string
  links: NavigationLink[]
  cta?: NavigationCTA
  width?: 'sm' | 'md' | 'lg'
}

export interface LinkGroupSection {
  type: 'linkGroup'
  links: Array<{
    label: string
    href: string
    badge?: string
    disabled?: boolean
  }>
}

export interface SeparatorSection {
  type: 'separator'
}

export type NavigationSection = DropdownSection | LinkGroupSection | SeparatorSection

export interface NavigationActions {
  docs?: {
    url: string
    label?: string
  }
  github?: {
    url: string
  }
  cta?: {
    label: string
    href?: string
    onClick?: () => void
    ariaLabel?: string
  }
}

export interface NavigationConfig {
  sections: NavigationSection[]
  actions: NavigationActions
  /** Optional href for the logo - if provided, logo becomes clickable */
  logoHref?: string
}
