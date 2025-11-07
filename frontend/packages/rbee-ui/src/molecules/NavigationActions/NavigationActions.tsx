'use client'

// TEAM-460: NavigationActions molecule - right-side actions for navigation
// Atomic Design: MOLECULE (combines atoms: Button, IconButton, ThemeToggle)

import { Button } from '@rbee/ui/atoms/Button'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import { GitHubIcon } from '@rbee/ui/icons'
import { ThemeToggle } from '@rbee/ui/molecules'
import { BookOpen } from 'lucide-react'
import Link from 'next/link'

export interface NavigationActionsProps {
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

export function NavigationActions({ docs, github, cta }: NavigationActionsProps) {
  return (
    <div className="flex items-center gap-3 justify-self-end">
      {docs && (
        <Button
          variant="ghost"
          size="sm"
          className="hidden md:flex h-9 px-2 gap-1 text-muted-foreground hover:text-foreground"
          asChild
        >
          <Link href={docs.url} target="_blank" rel="noopener">
            <BookOpen className="size-4" />
            {docs.label || 'Docs'}
          </Link>
        </Button>
      )}

      <div className="flex items-center gap-1 rounded-md p-0.5 bg-muted/40 ring-1 ring-border/60 shadow-[inset_0_0_0_1px_var(--border)]">
        {github && (
          <IconButton asChild aria-label="Open rbee on GitHub" title="GitHub">
            <a
              href={github.url}
              target="_blank"
              rel="noopener noreferrer"
              className="motion-safe:hover:animate-pulse"
            >
              <GitHubIcon size={20} />
            </a>
          </IconButton>
        )}

        <ThemeToggle />
      </div>

      {cta && (
        <Button
          className="hidden md:flex bg-primary hover:bg-primary/85 text-primary-foreground h-9"
          aria-label={cta.ariaLabel || cta.label}
          asChild={!!cta.href}
          onClick={cta.onClick}
        >
          {cta.href ? <Link href={cta.href}>{cta.label}</Link> : <span>{cta.label}</span>}
        </Button>
      )}
    </div>
  )
}
