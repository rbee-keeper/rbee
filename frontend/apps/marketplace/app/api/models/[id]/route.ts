// TEAM-405: Single model API route
//! Fetch specific model details via marketplace-node

import { NextRequest, NextResponse } from 'next/server'

const MARKETPLACE_API_URL = process.env.MARKETPLACE_API_URL || 'http://localhost:3001'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
  const modelId = decodeURIComponent(id)
  
  try {
    // Call marketplace-node API
    const response = await fetch(
      `${MARKETPLACE_API_URL}/api/models/${encodeURIComponent(modelId)}`,
      {
        headers: {
          'Accept': 'application/json',
        },
        // Cache for 1 hour
        next: { revalidate: 3600 }
      }
    )
    
    if (!response.ok) {
      throw new Error(`Marketplace API error: ${response.statusText}`)
    }
    
    const model = await response.json()
    
    return NextResponse.json(model, {
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=7200'
      }
    })
  } catch (error) {
    console.error(`Error fetching model ${modelId}:`, error)
    
    // Fallback to HuggingFace API
    try {
      const hfResponse = await fetch(
        `https://huggingface.co/api/models/${modelId}`,
        {
          headers: {
            'Accept': 'application/json',
          }
        }
      )
      
      if (!hfResponse.ok) {
        throw new Error('HuggingFace API also failed')
      }
      
      const model = await hfResponse.json()
      
      return NextResponse.json(model, {
        headers: {
          'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=7200'
        }
      })
    } catch (fallbackError) {
      return NextResponse.json(
        { error: 'Model not found' },
        { status: 404 }
      )
    }
  }
}
