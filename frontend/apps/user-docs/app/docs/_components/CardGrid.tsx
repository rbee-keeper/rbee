interface CardGridProps {
  columns?: 2 | 3 | 4
  children: React.ReactNode
}

const gridCols = {
  2: 'md:grid-cols-2',
  3: 'md:grid-cols-3',
  4: 'md:grid-cols-4',
}

export function CardGrid({ columns = 2, children }: CardGridProps) {
  return (
    <div className={`grid gap-4 my-6 ${gridCols[columns]}`}>
      {children}
    </div>
  )
}
