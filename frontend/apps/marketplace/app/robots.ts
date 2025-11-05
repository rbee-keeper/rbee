// TEAM-410: Robots.txt for SEO
import { MetadataRoute } from 'next'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: '/api/',
    },
    sitemap: 'https://marketplace.rbee.dev/sitemap.xml',
  }
}
