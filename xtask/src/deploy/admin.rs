// Deploy admin to Cloudflare Pages (SSR with @opennextjs/cloudflare)
// TEAM-480: Admin uses SSR (Clerk auth, Stripe, KV) - deployed via npm run deploy

use anyhow::{Context, Result};
use std::process::Command;

pub fn deploy(production: bool, dry_run: bool) -> Result<()> {
    let env = if production { "PRODUCTION" } else { "PREVIEW" };
    println!("🚀 Deploying Admin App to Cloudflare Pages (SSR) - {}", env);
    println!("🔐 Server-side rendering with Clerk auth, Stripe, KV storage");
    println!();

    let app_dir = "frontend/apps/admin";
    let domain = if production {
        "backend.rbee.dev"
    } else {
        "backend-dev.rbee.dev"
    };

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && npm run deploy", app_dir);
        println!("  Target: {}", domain);
        return Ok(());
    }

    // Deploy using @opennextjs/cloudflare (handles SSR build + deploy)
    println!("🔨 Building and deploying with @opennextjs/cloudflare...");
    let status = Command::new("npm")
        .args(&["run", "deploy"])
        .current_dir(app_dir)
        .status()
        .context("Failed to run npm run deploy")?;

    if !status.success() {
        anyhow::bail!("Deployment failed");
    }

    println!();
    println!("✅ Admin deployed successfully!");
    println!("📍 URL: https://{}", domain);
    println!();

    Ok(())
}
