// TEAM-476: Marketplace app layout with navigation
import type { Metadata } from 'next'
import { MarketplaceNav } from '@/components/MarketplaceNav'
import './globals.css'

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
