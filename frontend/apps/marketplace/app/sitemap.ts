// TEAM-410: Sitemap generation for SEO
import { MetadataRoute } from 'next'
import { listHuggingFaceModels } from '@rbee/marketplace-node'
import { modelIdToSlug } from '@/lib/slugify'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  // TEAM-457: Use environment variable with production fallback
  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://marketplace.rbee.dev'
  
  let modelUrls: MetadataRoute.Sitemap = []
  
  try {
    // Fetch all models for sitemap
    const models = await listHuggingFaceModels({ limit: 100 })
    
    // Generate model URLs
    modelUrls = models.map((model) => ({
      url: `${baseUrl}/models/${modelIdToSlug(model.id)}`,
      lastModified: model.lastModified ? new Date(model.lastModified) : new Date(),
      changeFrequency: 'weekly' as const,
      priority: 0.8,
    }))
  } catch (error) {
    console.error('Failed to fetch models for sitemap:', error)
    // Return basic sitemap if model fetch fails
  }
  
  return [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${baseUrl}/models`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
    ...modelUrls,
  ]
}
