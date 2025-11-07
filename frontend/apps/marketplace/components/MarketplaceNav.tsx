'use client'

import { BrandLogo, ThemeToggle } from '@rbee/ui/molecules'
import { Button } from '@rbee/ui/atoms/Button'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import { GitHubIcon } from '@rbee/ui/icons'
import { BookOpen } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export function MarketplaceNav() {
  const pathname = usePathname()

  return (
    <>
      {/* Skip to content link */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[60] rounded-md bg-primary px-3 py-2 text-primary-foreground shadow"
      >
        Skip to content
      </a>

      <nav
        aria-label="Primary"
        className="fixed top-0 inset-x-0 z-50 bg-background/95 supports-[backdrop-filter]:bg-background/70 backdrop-blur-sm border-b border-border/60"
      >
        <div className="relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/15 before:to-transparent">
          <div className="px-4 sm:px-6 lg:px-8 mx-auto max-w-7xl">
            <div className="grid grid-cols-[auto_1fr_auto] items-center h-16 md:h-14">
              {/* Zone A: Logo + Brand */}
              <Link href="/" aria-label="rbee home">
                <BrandLogo priority />
              </Link>

              {/* Zone B: Navigation Links (Desktop) - TEAM-457: Fixed marketplace navigation */}
              <div className="hidden md:flex items-center justify-center gap-6 font-sans">
                {/* Models Group */}
                <div className="flex items-center gap-4">
                  <Link
                    href="/models"
                    className={`text-sm font-medium transition-colors hover:text-foreground ${
                      pathname === '/models' ? 'text-foreground' : 'text-foreground/80'
                    }`}
                    aria-current={pathname === '/models' ? 'page' : undefined}
                  >
                    LLM Models
                  </Link>
                  <Link
                    href="/models?type=sd"
                    className="text-sm font-medium text-foreground/60 hover:text-foreground/80 transition-colors cursor-not-allowed"
                    onClick={(e) => e.preventDefault()}
                    title="Coming soon"
                  >
                    SD Models
                    <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Soon</span>
                  </Link>
                </div>

                {/* Separator */}
                <div className="h-4 w-px bg-border" />

                {/* Workers Group */}
                <div className="flex items-center gap-4">
                  <Link
                    href="/workers"
                    className={`text-sm font-medium transition-colors hover:text-foreground ${
                      pathname === '/workers' ? 'text-foreground' : 'text-foreground/80'
                    }`}
                    aria-current={pathname === '/workers' ? 'page' : undefined}
                  >
                    LLM Workers
                  </Link>
                  <Link
                    href="/workers?type=image"
                    className="text-sm font-medium text-foreground/60 hover:text-foreground/80 transition-colors cursor-not-allowed"
                    onClick={(e) => e.preventDefault()}
                    title="Coming soon"
                  >
                    Image Workers
                    <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">Soon</span>
                  </Link>
                </div>
              </div>

              {/* Zone C: Actions (Desktop) */}
              <div className="flex items-center gap-3 justify-self-end">
                <Button
                  variant="ghost"
                  size="sm"
                  className="hidden md:flex h-9 px-2 gap-1 text-muted-foreground hover:text-foreground"
                  asChild
                >
                  <Link href="https://github.com/veighnsche/llama-orch/tree/main/docs" target="_blank" rel="noopener">
                    <BookOpen className="size-4" />
                    Docs
                  </Link>
                </Button>

                <div className="flex items-center gap-1 rounded-md p-0.5 bg-muted/40 ring-1 ring-border/60 shadow-[inset_0_0_0_1px_var(--border)]">
                  <IconButton asChild aria-label="Open rbee on GitHub" title="GitHub">
                    <a
                      href="https://github.com/veighnsche/llama-orch"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="motion-safe:hover:animate-pulse"
                    >
                      <GitHubIcon size={20} />
                    </a>
                  </IconButton>

                  <ThemeToggle />
                </div>

                <Button
                  className="hidden md:flex bg-primary hover:bg-primary/85 text-primary-foreground h-9"
                  aria-label="Back to rbee.dev"
                  asChild
                >
                  <Link href="https://rbee.dev">
                    Back to rbee.dev
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </>
  )
}
