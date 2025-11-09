// TEAM-405: Models API route
//! Connects to marketplace-node backend for dynamic search/filtering

import { type NextRequest, NextResponse } from 'next/server'

const MARKETPLACE_API_URL = process.env.MARKETPLACE_API_URL || 'http://localhost:3001'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams

  // Extract query parameters
  const query = searchParams.get('query') || ''
  const sort = searchParams.get('sort') || 'downloads'
  const filterTags = searchParams.get('filterTags')?.split(',').filter(Boolean) || []
  const limit = parseInt(searchParams.get('limit') || '50')

  try {
    // Call marketplace-node API (which uses marketplace-sdk)
    const response = await fetch(
      `${MARKETPLACE_API_URL}/api/models?${new URLSearchParams({
        query,
        sort,
        filterTags: filterTags.join(','),
        limit: limit.toString(),
      })}`,
      {
        headers: {
          Accept: 'application/json',
        },
        // Cache for 5 minutes
        next: { revalidate: 300 },
      },
    )

    if (!response.ok) {
      throw new Error(`Marketplace API error: ${response.statusText}`)
    }

    const models = await response.json()

    return NextResponse.json(models, {
      headers: {
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
      },
    })
  } catch (error) {
    console.error('Error fetching models:', error)

    // Fallback to HuggingFace API if marketplace-node is down
    try {
      const hfResponse = await fetch(
        `https://huggingface.co/api/models?sort=${sort}&direction=-1&limit=${limit}&filter=text-generation${query ? `&search=${query}` : ''}`,
        {
          headers: {
            Accept: 'application/json',
          },
        },
      )

      if (!hfResponse.ok) {
        throw new Error('HuggingFace API also failed')
      }

      const models = await hfResponse.json()

      return NextResponse.json(models, {
        headers: {
          'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
        },
      })
    } catch {
      return NextResponse.json({ error: 'Failed to fetch models' }, { status: 500 })
    }
  }
}
