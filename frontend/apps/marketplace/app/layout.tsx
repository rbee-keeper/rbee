import type { Metadata } from 'next'
import type React from 'react'
// 🚨 TURBOREPO PATTERN: Import app CSS (with JIT scanning), then UI CSS (pre-built tokens)
// ✅ App CSS: Enables arbitrary values like translate-y-[2rem] in app components
// ✅ UI CSS: Provides design tokens and component styles
// ✅ All fonts are loaded in @rbee/ui/styles.css (Geist Sans, Geist Mono, Source Serif 4)
import './globals.css'
import '@rbee/ui/styles.css'
import { Footer } from '@rbee/ui/organisms'
import { ThemeProvider } from 'next-themes'
import { Suspense } from 'react'
import { MarketplaceNav } from '@/components/MarketplaceNav'

// TEAM-427: Always set metadataBase to production URL to avoid warnings
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://marketplace.rbee.dev'

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'rbee Model Marketplace - AI Language Models',
    template: '%s | rbee Marketplace',
  },
  description:
    'Browse and discover AI language models for your projects. Pre-rendered static pages for optimal SEO and performance.',
  keywords: ['AI', 'language models', 'LLM', 'machine learning', 'marketplace'],
  authors: [{ name: 'rbee' }],
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://marketplace.rbee.dev',
    siteName: 'rbee Model Marketplace',
    title: 'rbee Model Marketplace - AI Language Models',
    description: 'Browse and discover AI language models for your projects',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Model Marketplace',
    description: 'Browse and discover AI language models',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-serif bg-background text-foreground">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <MarketplaceNav />
          <main id="main" className="pt-16 md:pt-14 min-h-screen">
            <Suspense fallback={null}>{children}</Suspense>
          </main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  )
}
