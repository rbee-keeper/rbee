// TEAM-410: Robots.txt for SEO
import { MetadataRoute } from 'next'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: '/api/',
    },
    // TEAM-457: Use environment variable with production fallback
    sitemap: `${process.env.NEXT_PUBLIC_SITE_URL || 'https://marketplace.rbee.dev'}/sitemap.xml`,
  }
}
