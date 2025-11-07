'use client'

// TEAM-460: LinkGroup molecule - simple link group for marketplace
// Atomic Design: MOLECULE (combines atoms: Link)

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export interface LinkGroupProps {
  links: Array<{
    label: string
    href: string
    badge?: string
    disabled?: boolean
  }>
}

export function LinkGroup({ links }: LinkGroupProps) {
  const pathname = usePathname()

  return (
    <div className="flex items-center gap-4">
      {links.map((link, index) => {
        const key = link.href || `link-${index}`
        
        if (link.disabled) {
          return (
            <span
              key={key}
              className="text-sm font-medium text-foreground/60 cursor-not-allowed flex items-center gap-1.5"
            >
              {link.label}
              {link.badge && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                  {link.badge}
                </span>
              )}
            </span>
          )
        }

        return (
          <Link
            key={key}
            href={link.href}
            className={`text-sm font-medium transition-colors hover:text-foreground ${
              pathname === link.href ? 'text-foreground' : 'text-foreground/80'
            }`}
            aria-current={pathname === link.href ? 'page' : undefined}
          >
            {link.label}
          </Link>
        )
      })}
    </div>
  )
}
