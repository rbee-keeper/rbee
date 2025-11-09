import type { Metadata } from 'next'
import { getPageMap } from 'nextra/page-map'
import { Layout } from 'nextra-theme-docs'
// Import order: app CSS (JIT scanning) → UI CSS (tokens) → Nextra theme
import './globals.css'
import '@rbee/ui/styles.css'
import 'nextra-theme-docs/style.css'
import { Navigation } from '@/components/Navigation'

// TEAM-427: Always set metadataBase to production URL to avoid warnings
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://docs.rbee.dev'

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
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

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const pageMap = await getPageMap('/')

  const footerContent = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
      <span>{new Date().getFullYear()} © rbee. Your private AI cloud, in one command.</span>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <a key="github" href="https://github.com/veighnsche/llama-orch" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>
        <a key="website" href="https://rbee.dev" target="_blank" rel="noopener noreferrer">
          rbee.dev
        </a>
      </div>
    </div>
  )

  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Navigation />
        <main id="main">
          <Layout
            pageMap={pageMap}
            docsRepositoryBase="https://github.com/veighnsche/llama-orch/tree/main/frontend/apps/user-docs/app"
            sidebar={{
              defaultMenuCollapseLevel: 1,
              toggleButton: true,
            }}
            footer={footerContent}
          >
            {children}
          </Layout>
        </main>
      </body>
    </html>
  )
}
