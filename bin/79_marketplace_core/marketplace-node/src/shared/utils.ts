// TEAM-XXX: Shared utility functions
// Only pure utilities with no provider-specific logic belong here

/**
 * Format bytes to human-readable string
 * 
 * @param bytes - Number of bytes
 * @returns Formatted string (e.g., "1.5 GB")
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}
