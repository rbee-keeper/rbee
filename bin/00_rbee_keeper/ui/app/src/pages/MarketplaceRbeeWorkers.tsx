// TEAM-405: Marketplace Rbee Workers page
// Browse and install rbee worker binaries (placeholder for future implementation)

export function MarketplaceRbeeWorkers() {
  return (
    <div className="h-full flex flex-col items-center justify-center p-6">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-bold">Rbee Workers</h1>
        <p className="text-muted-foreground">
          Worker catalog integration coming soon. This will allow you to browse and install rbee worker binaries.
        </p>
        <div className="text-sm text-muted-foreground">
          <p>Planned features:</p>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Browse available worker types (CPU, CUDA, Metal)</li>
            <li>Filter by platform and architecture</li>
            <li>Install workers to hives</li>
            <li>View worker versions and changelogs</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
