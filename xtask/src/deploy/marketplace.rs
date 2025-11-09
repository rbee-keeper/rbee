// Deploy marketplace to Cloudflare Pages (SSG)
// TEAM-427: Updated to use Next.js static export (out/ directory)
// TEAM-463: Refactored to use nextjs_ssg abstraction

use anyhow::Result;
use super::nextjs_ssg::{deploy_nextjs_ssg, NextJsDeployConfig};

pub fn deploy(production: bool, dry_run: bool) -> Result<()> {
    println!("📄 All 455 pages pre-rendered at build time");
    println!("   - 200 model redirect pages (/models/[slug])");
    println!("   - 200 model detail pages (100 HF + 100 CivitAI)");
    println!("   - 55 other pages (filters, workers, etc.)");
    println!();

    let config = NextJsDeployConfig {
        app_name: "Marketplace",
        app_dir: "frontend/apps/marketplace",
        project_name: "rbee-marketplace",
        production_url: "https://rbee-marketplace.pages.dev",
        custom_domain: Some("https://marketplace.rbee.dev"),
        env_vars: vec![
            ("MARKETPLACE_API_URL", "https://gwc.rbee.dev"),
            ("NEXT_PUBLIC_SITE_URL", "https://marketplace.rbee.dev"),
            ("NEXT_DISABLE_DEVTOOLS", "1"),
        ],
    };

    deploy_nextjs_ssg(config, production, dry_run)
}
