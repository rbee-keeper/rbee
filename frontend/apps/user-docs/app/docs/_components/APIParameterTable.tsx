'use client'

import { Badge, Input, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@rbee/ui/atoms'
import { Search } from 'lucide-react'
import { useState } from 'react'

export interface APIParameter {
  name: string
  type: string
  required: boolean
  default?: string
  description: string
}

interface APIParameterTableProps {
  parameters: APIParameter[]
  searchable?: boolean
}

export function APIParameterTable({ parameters, searchable = true }: APIParameterTableProps) {
  const [search, setSearch] = useState('')

  const filtered = parameters.filter(
    (p) =>
      search === '' ||
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.description.toLowerCase().includes(search.toLowerCase()) ||
      p.type.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="my-6">
      {searchable && parameters.length > 5 && (
        <div className="mb-4 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search parameters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
      )}
      <div className="border rounded-lg overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[200px]">Parameter</TableHead>
              <TableHead className="w-[120px]">Type</TableHead>
              <TableHead className="w-[100px]">Required</TableHead>
              <TableHead className="w-[120px]">Default</TableHead>
              <TableHead>Description</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
                  No parameters found
                </TableCell>
              </TableRow>
            ) : (
              filtered.map((param) => (
                <TableRow key={param.name}>
                  <TableCell className="font-mono font-semibold">{param.name}</TableCell>
                  <TableCell>
                    <code className="text-sm bg-muted px-2 py-1 rounded">{param.type}</code>
                  </TableCell>
                  <TableCell>
                    {param.required ? (
                      <Badge variant="destructive" className="text-xs">
                        Required
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">
                        Optional
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-sm text-muted-foreground">{param.default || '—'}</TableCell>
                  <TableCell className="text-sm">{param.description}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
      {filtered.length > 0 && filtered.length < parameters.length && (
        <p className="text-sm text-muted-foreground mt-2">
          Showing {filtered.length} of {parameters.length} parameters
        </p>
      )}
    </div>
  )
}
