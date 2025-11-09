import { getPageMap } from 'nextra/page-map'
import { Layout } from 'nextra-theme-docs'

export default async function DocsLayout({ children }: { children: React.ReactNode }) {
  const pageMap = await getPageMap('/docs')

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
  )
}
