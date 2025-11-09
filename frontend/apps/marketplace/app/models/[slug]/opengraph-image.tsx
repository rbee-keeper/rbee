// TEAM-417: Generate dynamic OG image for each model
// Creates unique social sharing images for model detail pages

import { listHuggingFaceModels } from '@rbee/marketplace-node'
import { ImageResponse } from 'next/og'

// TEAM-417: Use Node.js runtime (not Edge) because marketplace-node uses WASM with fs
export const runtime = 'nodejs'
export const alt = 'Model Detail - rbee Marketplace'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

export default async function Image({ params }: { params: Promise<{ slug: string }> }) {
  // TEAM-417: Await params (Next.js 15 requirement)
  const { slug } = await params

  // TEAM-417: Fetch model data for the specific slug
  const models = await listHuggingFaceModels({ limit: 1000 })
  const model = models.find((m) => m.id === slug)

  // TEAM-417: Extract model name and author
  const modelName = model?.name || slug
  const modelAuthor = model?.author || 'Unknown'

  return new ImageResponse(
    <div
      style={{
        fontSize: 64,
        background: 'linear-gradient(to bottom, #1e293b, #0f172a)',
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        padding: '60px',
      }}
    >
      {/* TEAM-417: rbee branding */}
      <div
        style={{
          fontSize: 48,
          marginBottom: 30,
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
        }}
      >
        <span>🐝</span>
        <span>rbee</span>
      </div>

      {/* TEAM-417: Model name (main focus) */}
      <div
        style={{
          fontSize: 72,
          fontWeight: 'bold',
          textAlign: 'center',
          marginBottom: 20,
          maxWidth: '90%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {modelName}
      </div>

      {/* TEAM-417: Model author */}
      <div
        style={{
          fontSize: 32,
          opacity: 0.8,
          marginBottom: 40,
        }}
      >
        by {modelAuthor}
      </div>

      {/* TEAM-417: Call to action */}
      <div
        style={{
          fontSize: 28,
          opacity: 0.7,
          borderTop: '2px solid rgba(255,255,255,0.3)',
          paddingTop: 30,
          marginTop: 20,
        }}
      >
        Run Locally with rbee
      </div>
    </div>,
    { ...size },
  )
}
