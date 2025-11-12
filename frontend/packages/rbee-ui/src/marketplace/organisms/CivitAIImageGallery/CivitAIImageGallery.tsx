// TEAM-463: Premium image gallery for CivitAI models
// TEAM-478: Enhanced with Carousel for thumbnail navigation
// Uses Next.js Image with fill for responsive images

'use client'

import { Button, Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import { ChevronLeft, ChevronRight, Maximize2 } from 'lucide-react'
import Image from 'next/image'
import { useState } from 'react'

export interface CivitAIImage {
  url: string
  nsfw: boolean
  width: number
  height: number
}

export interface CivitAIImageGalleryProps {
  images: CivitAIImage[]
  modelName: string
}

export function CivitAIImageGallery({ images, modelName }: CivitAIImageGalleryProps) {
  const [selectedIndex, setSelectedIndex] = useState(0)

  if (images.length === 0) {
    return (
      <div className="relative aspect-square overflow-hidden rounded-xl border bg-muted flex items-center justify-center">
        <p className="text-muted-foreground">No preview images available</p>
      </div>
    )
  }

  const selectedImage = images[selectedIndex]

  // TEAM-472: Guard check for noUncheckedIndexedAccess
  if (!selectedImage) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed bg-muted p-12">
        <p className="text-muted-foreground">No preview images available</p>
      </div>
    )
  }

  const handlePrevious = () => {
    setSelectedIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1))
  }

  const handleNext = () => {
    setSelectedIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1))
  }

  return (
    <div className="space-y-4">
      {/* Main Image */}
      <div className="group relative aspect-square overflow-hidden bg-muted shadow-lg">
        <Image
          src={selectedImage.url}
          alt={`${modelName} preview ${selectedIndex + 1}`}
          fill
          className="object-cover transition-transform duration-300 group-hover:scale-105"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 800px"
          priority={selectedIndex === 0}
        />

        {/* Navigation Arrows */}
        {images.length > 1 && (
          <>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              className="absolute left-4 top-1/2 -translate-y-1/2 opacity-0 transition-opacity group-hover:opacity-100 shadow-xl"
              onClick={handlePrevious}
            >
              <ChevronLeft className="size-5" />
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              className="absolute right-4 top-1/2 -translate-y-1/2 opacity-0 transition-opacity group-hover:opacity-100 shadow-xl"
              onClick={handleNext}
            >
              <ChevronRight className="size-5" />
            </Button>
          </>
        )}

        {/* Image Counter */}
        <div className="absolute bottom-4 right-4 rounded-full bg-black/60 px-3 py-1 text-xs text-white backdrop-blur-sm">
          {selectedIndex + 1} / {images.length}
        </div>

        {/* Fullscreen Button */}
        <Button
          type="button"
          variant="secondary"
          size="icon"
          className="absolute top-4 right-4 opacity-0 transition-opacity group-hover:opacity-100 shadow-xl"
        >
          <Maximize2 className="size-4" />
        </Button>
      </div>

      {/* Thumbnail Carousel - TEAM-478: Shows ALL images with swipe/scroll */}
      {images.length > 1 && (
        <Carousel
          opts={{
            align: 'start',
            loop: false,
            slidesToScroll: 1,
          }}
          className="w-full py-2"
        >
          <CarouselContent className="-ml-2 py-2">
            {images.map((image, idx) => (
              <CarouselItem key={`thumb-${image.url}-${idx}`} className="basis-1/5 pl-2">
                <button
                  type="button"
                  onClick={() => setSelectedIndex(idx)}
                  className={cn(
                    'group relative aspect-square overflow-hidden rounded-lg border-2 transition-all duration-200 w-full',
                    selectedIndex === idx
                      ? 'border-primary ring-4 ring-primary/20 scale-105'
                      : 'border-transparent hover:border-primary/50 hover:scale-105',
                  )}
                >
                  <Image
                    src={image.url}
                    alt={`Thumbnail ${idx + 1}`}
                    fill
                    className="object-cover transition-transform duration-200 group-hover:scale-110"
                    sizes="(max-width: 768px) 20vw, 150px"
                  />
                  {selectedIndex === idx && <div className="absolute inset-0 bg-primary/10" />}
                </button>
              </CarouselItem>
            ))}
          </CarouselContent>
          <CarouselPrevious className="-left-4" />
          <CarouselNext className="-right-4" />
        </Carousel>
      )}
    </div>
  )
}
