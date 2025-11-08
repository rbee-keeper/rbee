// Deployment commands
// Created by: TEAM-451
// Individual deployment commands for each app and binary

pub mod binaries;
pub mod commercial;
pub mod docs;
pub mod marketplace;
pub mod worker_catalog;

use anyhow::Result;

pub fn run(app: &str, dry_run: bool) -> Result<()> {
    match app {
        // Cloudflare deployments
        "worker" | "gwc" | "worker-catalog" => worker_catalog::deploy(dry_run),
        "commercial" => commercial::deploy(dry_run),
        "marketplace" => marketplace::deploy(dry_run),
        "docs" | "user-docs" => docs::deploy(dry_run),
        
        // Binary deployments (GitHub Releases)
        "keeper" | "rbee-keeper" => binaries::deploy_keeper(dry_run),
        "queen" | "queen-rbee" => binaries::deploy_queen(dry_run),
        "hive" | "rbee-hive" => binaries::deploy_hive(dry_run),
        "llm-worker" | "llm-worker-rbee" => binaries::deploy_llm_worker(dry_run),
        "sd-worker" | "sd-worker-rbee" => binaries::deploy_sd_worker(dry_run),
        
        _ => anyhow::bail!(
            "Unknown app: {}\n\nCloudflare Apps:\n  - worker (gwc.rbee.dev)\n  - commercial (rbee.dev)\n  - marketplace (marketplace.rbee.dev)\n  - docs (docs.rbee.dev)\n\nRust Binaries (GitHub Releases):\n  - keeper (rbee-keeper)\n  - queen (queen-rbee)\n  - hive (rbee-hive)\n  - llm-worker (llm-worker-rbee)\n  - sd-worker (sd-worker-rbee)",
            app
        ),
    }
}
