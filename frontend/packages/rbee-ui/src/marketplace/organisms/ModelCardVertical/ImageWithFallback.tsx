// TEAM-461: Client component for image error handling with automatic fallbacks
'use client'

import { Sparkles } from 'lucide-react'
import { useState } from 'react'

interface ImageWithFallbackProps {
  src: string
  alt: string
  className?: string
  fallbackImages?: string[]
}

/**
 * Image component with automatic fallback handling
 * Tries alternative image URLs when primary fails
 * Falls back to placeholder if all images fail
 */
export function ImageWithFallback({ src, alt, className, fallbackImages = [] }: ImageWithFallbackProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [hasError, setHasError] = useState(false)

  // All possible image sources (primary + fallbacks)
  const allSources = [src, ...fallbackImages]
  const currentSrc = allSources[currentIndex]

  const handleError = () => {
    // Try next fallback image
    if (currentIndex < allSources.length - 1) {
      setCurrentIndex(currentIndex + 1)
    } else {
      // All images failed, show placeholder
      setHasError(true)
    }
  }

  if (hasError) {
    // Fallback placeholder when all images fail
    return (
      <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-primary/10 via-background to-muted">
        <Sparkles className="size-16 text-primary/30" />
      </div>
    )
  }

  return <img src={currentSrc} alt={alt} className={className} loading="lazy" onError={handleError} />
}
