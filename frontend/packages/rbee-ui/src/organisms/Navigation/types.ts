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

// TEAM-463: Direct link in navbar (no dropdown)
export interface DirectLinkSection {
  type: 'directLink'
  label: string
  href: string
  icon?: LucideIcon | string
}

// TEAM-463: Two-column dropdown with support for multiple sections per column
export interface ColumnSection {
  title: string
  links: NavigationLink[]
}

export interface TwoColumnDropdownSection {
  type: 'twoColumnDropdown'
  title: string
  leftColumn: ColumnSection | ColumnSection[]
  rightColumn: ColumnSection | ColumnSection[]
  cta?: NavigationCTA
}

export type NavigationSection = DropdownSection | LinkGroupSection | SeparatorSection | DirectLinkSection | TwoColumnDropdownSection

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
