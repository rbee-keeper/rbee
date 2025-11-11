// TEAM-410: Robots.txt for SEO
// TEAM-468: Updated with optimal SEO configuration
import type { MetadataRoute } from 'next'

export const dynamic = 'force-static'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: [
        '/api/', // Block API endpoints
        '/_next/', // Block Next.js internals
        '/admin/', // Block admin paths
        '/*.json$', // Block JSON files for security
      ],
    },
    // TEAM-457: Use environment variable with production fallback
    sitemap: `${process.env.NEXT_PUBLIC_SITE_URL || 'https://marketplace.rbee.dev'}/sitemap.xml`,
  }
}
