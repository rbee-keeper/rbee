// TEAM-417: Generate Open Graph image for social sharing
// Base OG image for rbee Marketplace homepage

import { ImageResponse } from 'next/og'

// TEAM-417: Force static generation for static export
export const dynamic = 'force-static'
export const alt = 'rbee Marketplace - Run LLMs Locally'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

export default async function Image() {
  return new ImageResponse(
    <div
      style={{
        fontSize: 128,
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
      <div style={{ fontSize: 96, marginBottom: 40 }}>🐝</div>
      <div style={{ fontSize: 72, fontWeight: 'bold', marginBottom: 20 }}>rbee Marketplace</div>
      <div style={{ fontSize: 36, opacity: 0.9, textAlign: 'center' }}>Run LLMs Locally on Your Hardware</div>
    </div>,
    { ...size },
  )
}
