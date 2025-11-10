'use client'

// TEAM-460: NavigationDropdown molecule - extracted from Navigation organism
// Atomic Design: MOLECULE (combines atoms: NavigationMenu, Button, Link)
// NOTE: Different from atoms/DropdownMenu (Radix primitive)

import { Button } from '@rbee/ui/atoms/Button'
import {
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuTrigger,
} from '@rbee/ui/atoms/NavigationMenu'
import type { NavigationCTA, NavigationLink } from '@rbee/ui/organisms/Navigation'
import { BookOpen } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { renderIcon } from '../../utils/renderIcon'

export interface NavigationDropdownProps {
  title: string
  links: NavigationLink[]
  cta?: NavigationCTA
  width?: 'sm' | 'md' | 'lg'
}

const widthClasses = {
  sm: 'w-[280px]',
  md: 'w-[560px]',
  lg: 'w-[840px]',
}

export function NavigationDropdown({ title, links, cta, width = 'sm' }: NavigationDropdownProps) {
  const pathname = usePathname()
  const gridCols = width === 'lg' ? 'grid-cols-2' : 'grid-cols-1'

  return (
    <NavigationMenuItem>
      <NavigationMenuTrigger className="!bg-transparent hover:!bg-transparent focus:!bg-transparent data-[state=open]:!bg-transparent px-2 text-sm font-medium text-foreground/80 hover:!text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2">
        {title}
      </NavigationMenuTrigger>
      <NavigationMenuContent className="animate-fade-in md:motion-safe:animate-slide-in-down border border-border">
        <div className={`grid gap-1 p-3 ${widthClasses[width]}`}>
          <ul className={`grid gap-1 ${gridCols}`}>
            {links.map((link) => (
              <li key={link.href}>
                <NavigationMenuLink asChild>
                  <Link
                    href={link.href}
                    className={`${
                      link.description ? 'flex items-start gap-3' : 'block'
                    } select-none rounded-md p-2.5 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2`}
                    aria-current={pathname === link.href ? 'page' : undefined}
                  >
                    {link.icon && renderIcon(link.icon, { className: 'size-5 mt-0.5 shrink-0' })}
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
          {cta && (
            <div className="mt-2 flex items-center justify-between rounded bg-muted/30 p-2 ring-1 ring-border/50">
              <Link
                href="https://github.com/veighnsche/llama-orch/tree/main/docs"
                target="_blank"
                rel="noopener"
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                <BookOpen className="size-3.5" />
                Docs
              </Link>
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={cta.onClick} asChild={!!cta.href}>
                {cta.href ? <Link href={cta.href}>{cta.label}</Link> : <span>{cta.label}</span>}
              </Button>
            </div>
          )}
        </div>
      </NavigationMenuContent>
    </NavigationMenuItem>
  )
}
