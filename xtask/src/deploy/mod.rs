// Cloudflare deployment commands
// Created by: TEAM-451
// Individual deployment commands for each app

pub mod worker_catalog;
pub mod commercial;
pub mod marketplace;
pub mod docs;

use anyhow::Result;

pub fn run(app: &str, dry_run: bool) -> Result<()> {
    match app {
        "worker" | "gwc" | "worker-catalog" => worker_catalog::deploy(dry_run),
        "commercial" => commercial::deploy(dry_run),
        "marketplace" => marketplace::deploy(dry_run),
        "docs" | "user-docs" => docs::deploy(dry_run),
        _ => anyhow::bail!(
            "Unknown app: {}\n\nAvailable apps:\n  - worker (gwc.rbee.dev)\n  - commercial (rbee.dev)\n  - marketplace (marketplace.rbee.dev)\n  - docs (docs.rbee.dev)",
            app
        ),
    }
}
