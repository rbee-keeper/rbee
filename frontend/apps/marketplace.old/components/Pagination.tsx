// TEAM-427: Pagination using rbee-ui atoms (proper atomic design)
import {
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
  Pagination as PaginationRoot,
} from '@rbee/ui/atoms'

interface PaginationProps {
  currentPage: number
  totalPages: number
  baseUrl: string
}

/**
 * Pagination component using rbee-ui atoms
 * Follows atomic design pattern with proper composition
 */
export function Pagination({ currentPage, totalPages, baseUrl }: PaginationProps) {
  // Show max 7 page numbers with ellipsis
  const getPageNumbers = () => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1)
    }

    // Always show first page, last page, current page, and neighbors
    const pages: (number | 'ellipsis')[] = [1]

    if (currentPage > 3) {
      pages.push('ellipsis')
    }

    // Show current page and neighbors
    for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
      pages.push(i)
    }

    if (currentPage < totalPages - 2) {
      pages.push('ellipsis')
    }

    if (totalPages > 1) {
      pages.push(totalPages)
    }

    return pages
  }

  const pages = getPageNumbers()

  return (
    <PaginationRoot className="my-8">
      <PaginationContent>
        {/* Previous button */}
        {currentPage > 1 && (
          <PaginationItem>
            <PaginationPrevious href={`${baseUrl}?page=${currentPage - 1}`} />
          </PaginationItem>
        )}

        {/* Page numbers */}
        {pages.map((page, idx) =>
          page === 'ellipsis' ? (
            <PaginationItem key={`ellipsis-${idx}`}>
              <PaginationEllipsis />
            </PaginationItem>
          ) : (
            <PaginationItem key={page}>
              <PaginationLink href={`${baseUrl}?page=${page}`} isActive={page === currentPage}>
                {page}
              </PaginationLink>
            </PaginationItem>
          ),
        )}

        {/* Next button */}
        {currentPage < totalPages && (
          <PaginationItem>
            <PaginationNext href={`${baseUrl}?page=${currentPage + 1}`} />
          </PaginationItem>
        )}
      </PaginationContent>
    </PaginationRoot>
  )
}
