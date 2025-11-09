import type { Metadata } from 'next'
// Import order: app CSS (JIT scanning) → UI CSS (tokens) → Nextra theme
import './globals.css'
import '@rbee/ui/styles.css'
import 'nextra-theme-docs/style.css'
import { Navigation } from '@/components/Navigation'

export const metadata: Metadata = {
  title: {
    default: 'rbee Documentation',
    template: '%s – rbee Docs',
  },
  description:
    'Technical documentation for rbee, the self-hosted AI orchestration layer. Learn how to deploy colonies, connect heterogeneous hardware, and use premium modules for routing, telemetry, and GDPR-focused auditing.',
  keywords: [
    'rbee',
    'AI orchestration',
    'self-hosted AI',
    'LLM hosting',
    'GPU management',
    'distributed AI',
    'GDPR compliance',
  ],
  authors: [{ name: 'Vince Liem', url: 'https://www.linkedin.com/in/vincepaulliem/' }],
  creator: 'rbee',
  publisher: 'rbee',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://docs.rbee.dev',
    title: 'rbee Documentation',
    description: 'Technical documentation for rbee, the self-hosted AI orchestration system.',
    siteName: 'rbee Documentation',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Documentation',
    description: 'Technical documentation for rbee, the self-hosted AI orchestration system.',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Navigation />
        <main id="main">{children}</main>
      </body>
    </html>
  )
}
