// TEAM-463: Two-column dropdown component for navbar
'use client'

import {
  NavigationMenuItem,
  NavigationMenuTrigger,
  NavigationMenuContent,
  NavigationMenuLink,
} from '@rbee/ui/atoms/NavigationMenu'
import { Button } from '@rbee/ui/atoms/Button'
import Link from 'next/link'
import * as LucideIcons from 'lucide-react'
import type { TwoColumnDropdownSection } from '@rbee/ui/organisms/Navigation/types'

export interface TwoColumnDropdownProps extends Omit<TwoColumnDropdownSection, 'type'> {}

export function TwoColumnDropdown({ title, leftColumn, rightColumn, cta }: TwoColumnDropdownProps) {
  const getIcon = (iconName?: string) => {
    if (!iconName) return null
    const Icon = (LucideIcons as any)[iconName]
    return Icon ? <Icon className="size-5 mt-0.5 shrink-0" /> : null
  }

  const renderSection = (section: any) => (
    <div key={section.title} className="mb-4 last:mb-0">
      <h3 className="text-sm font-semibold mb-3 text-foreground/90">{section.title}</h3>
      <ul className="grid gap-1">
        {section.links.map((link: any) => (
          <li key={link.href}>
            <NavigationMenuLink asChild>
              <Link
                href={link.href}
                className="flex items-start gap-3 select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2"
              >
                {getIcon(link.icon as string)}
                <div>
                  <div className="text-sm font-medium leading-none mb-1">{link.label}</div>
                  {link.description && (
                    <p className="text-[13px] leading-[1.2] text-muted-foreground">{link.description}</p>
                  )}
                </div>
              </Link>
            </NavigationMenuLink>
          </li>
        ))}
      </ul>
    </div>
  )

  const leftSections = Array.isArray(leftColumn) ? leftColumn : [leftColumn]
  const rightSections = Array.isArray(rightColumn) ? rightColumn : [rightColumn]

  return (
    <NavigationMenuItem>
      <NavigationMenuTrigger className="bg-transparent hover:bg-transparent focus:bg-transparent data-[state=open]:bg-transparent px-2 text-sm font-medium text-foreground/80 hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
        {title}
      </NavigationMenuTrigger>
      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
        <div className="grid grid-cols-2 gap-4 p-4 w-[720px]">
          {/* Left Column */}
          <div>
            {leftSections.map((section) => renderSection(section))}
          </div>

          {/* Right Column */}
          <div>
            {rightSections.map((section) => renderSection(section))}
          </div>
        </div>

        {/* CTA at bottom */}
        {cta && (
          <div className="border-t border-border p-3 bg-muted/30">
            <div className="flex items-center justify-end">
              {cta.href ? (
                <Button variant="ghost" size="sm" className="h-7 text-xs" asChild>
                  <Link href={cta.href}>{cta.label}</Link>
                </Button>
              ) : (
                <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={cta.onClick}>
                  {cta.label}
                </Button>
              )}
            </div>
          </div>
        )}
      </NavigationMenuContent>
    </NavigationMenuItem>
  )
}
