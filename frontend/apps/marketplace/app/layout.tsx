// TEAM-476: Marketplace app layout with navigation
import type { Metadata } from 'next'
import type React from 'react'
// 🚨 TURBOREPO PATTERN: Import app CSS (with JIT scanning), then UI CSS (pre-built tokens)
// ✅ App CSS: Enables arbitrary values like translate-y-[2rem] in app components
// ✅ UI CSS: Provides design tokens and component styles
// ✅ All fonts are loaded in @rbee/ui/styles.css (Geist Sans, Geist Mono, Source Serif 4)
import './globals.css'
import '@rbee/ui/styles.css'
import { MarketplaceNav } from '@/components/MarketplaceNav'

export const metadata: Metadata = {
  title: 'rbee Marketplace - AI Models & Workers',
  description: 'Browse and download AI models from HuggingFace and CivitAI. Find workers for your rbee cluster.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className="antialiased">
        <MarketplaceNav />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  )
}
