'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { DiscordIcon, GitHubIcon, XTwitterIcon } from '@rbee/ui/icons'
import { BrandLogo, FooterColumn } from '@rbee/ui/molecules'
import { MessageCircle } from 'lucide-react'

export function Footer() {
  const handleNewsletterSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    // TODO: Implement newsletter subscription
    const formData = new FormData(e.currentTarget)
    const email = formData.get('email')
    console.log('Newsletter signup:', email)
  }

  return (
    <footer className="relative bg-background text-muted-foreground pt-14 pb-10 before:absolute before:inset-x-0 before:top-0 before:h-px">
      <div className="container mx-auto px-4 space-y-12">
        {/* 1. Utility Bar */}
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between animate-fade-in">
          <div className="flex-1 max-w-2xl">
            <h3 className="text-sm font-semibold text-foreground mb-1">Stay in the loop</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Releases, roadmap, and self-hosting tips. 1–2 emails/month.
            </p>
            <form onSubmit={handleNewsletterSubmit} className="flex gap-2 max-w-md">
              <Input
                type="email"
                name="email"
                required
                placeholder="you@company.com"
                aria-label="Email address"
                className="h-10"
              />
              <Button type="submit" className="h-10 shrink-0">
                Subscribe
              </Button>
            </form>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline" size="sm" className="h-9">
              <a href="https://docs.rbee.dev">Documentation</a>
            </Button>
            <Button asChild variant="outline" size="sm" className="h-9">
              <a
                href="https://github.com/rbee-keeper/rbee"
                target="_blank"
                rel="noreferrer"
                title="Opens in a new tab"
              >
                Star on GitHub
              </a>
            </Button>
            <Button asChild size="sm" className="h-9">
              <a href="https://discord.gg/rbee" target="_blank" rel="noreferrer" title="Opens in a new tab">
                Join Discord
              </a>
            </Button>
          </div>
        </div>

        {/* 2. Sitemap Grid */}
        <nav aria-label="Footer" className="grid gap-8 md:grid-cols-5 animate-fade-in">
          <FooterColumn
            title="Product"
            links={[
              { href: '/features', text: 'Features' },
              { href: '/pricing', text: 'Pricing' },
              { href: '/security', text: 'Security' },
              { href: '/use-cases', text: 'Use Cases' },
              { href: '/compare', text: 'Compare' },
              {
                href: 'https://github.com/rbee-keeper/rbee/releases',
                text: 'Changelog',
                external: true,
              },
            ]}
          />

          <FooterColumn
            title="Solutions"
            links={[
              { href: '/use-cases/homelab', text: 'For Homelabs' },
              { href: '/developers', text: 'For Developers' },
              { href: '/enterprise', text: 'For Enterprise' },
              { href: '/use-cases/academic', text: 'For Academia' },
              { href: '/earn', text: 'Earn with Your GPUs' },
            ]}
          />

          <FooterColumn
            title="Resources"
            links={[
              {
                href: 'https://docs.rbee.dev',
                text: 'Documentation',
                external: true,
              },
              {
                href: 'https://docs.rbee.dev/getting-started',
                text: 'Getting Started',
                external: true,
              },
              {
                href: 'https://marketplace.rbee.dev',
                text: 'Model Marketplace',
                external: true,
              },
              { href: '/blog', text: 'Blog' },
              {
                href: 'https://github.com/rbee-keeper/rbee',
                text: 'GitHub',
                external: true,
              },
            ]}
          />

          <FooterColumn
            title="Community"
            links={[
              {
                href: 'https://discord.gg/rbee',
                text: 'Discord',
                external: true,
              },
              {
                href: 'https://github.com/rbee-keeper/rbee/discussions',
                text: 'Discussions',
                external: true,
              },
              { href: '/community', text: 'Community Hub' },
              {
                href: 'https://x.com/rbee',
                text: 'X (Twitter)',
                external: true,
              },
            ]}
          />

          <FooterColumn
            title="Company"
            links={[
              { href: '/about', text: 'About' },
              { href: '/contact', text: 'Contact' },
              { href: '/legal/privacy', text: 'Privacy' },
              { href: '/legal/terms', text: 'Terms' },
              {
                href: 'https://github.com/rbee-keeper/rbee/blob/main/LICENSE',
                text: 'License',
                external: true,
              },
            ]}
          />
        </nav>

        {/* 3. Bottom Bar */}
        <div className="border-t border-border pt-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between animate-fade-in-up">
          <div className="flex items-center gap-3">
            <BrandLogo size="sm" />
            <p className="text-sm text-muted-foreground">
              © 2025 rbee. Built with 🍯 by AI developers, for developers working with AI.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com/rbee-keeper/rbee"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <GitHubIcon size={20} />
            </a>
            <a
              href="https://github.com/rbee-keeper/rbee/discussions"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub Discussions"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <MessageCircle className="size-5" />
            </a>
            <a
              href="https://x.com/rbee"
              target="_blank"
              rel="noreferrer"
              aria-label="X (Twitter)"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <XTwitterIcon size={20} />
            </a>
            <a
              href="https://discord.gg/rbee"
              target="_blank"
              rel="noreferrer"
              aria-label="Discord"
              title="Opens in a new tab"
              className="rounded-md p-2 hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <DiscordIcon size={20} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
