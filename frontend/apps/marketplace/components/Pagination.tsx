// TEAM-462: Pagination component for marketplace pages
import Link from 'next/link'

interface PaginationProps {
  currentPage: number
  totalPages: number
  baseUrl: string
}

export function Pagination({ currentPage, totalPages, baseUrl }: PaginationProps) {
  return (
    <div className="flex items-center justify-center gap-2 my-8">
      {/* Previous button */}
      {currentPage > 1 && (
        <Link
          href={`${baseUrl}?page=${currentPage - 1}`}
          className="px-4 py-2 border rounded hover:bg-muted transition-colors"
        >
          ← Previous
        </Link>
      )}
      
      {/* Page numbers */}
      {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
        <Link
          key={page}
          href={`${baseUrl}?page=${page}`}
          className={`px-4 py-2 border rounded transition-colors ${
            page === currentPage 
              ? 'bg-primary text-primary-foreground font-bold' 
              : 'hover:bg-muted'
          }`}
        >
          {page}
        </Link>
      ))}
      
      {/* Next button */}
      {currentPage < totalPages && (
        <Link
          href={`${baseUrl}?page=${currentPage + 1}`}
          className="px-4 py-2 border rounded hover:bg-muted transition-colors"
        >
          Next →
        </Link>
      )}
    </div>
  )
}
