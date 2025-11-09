// Deploy worker catalog to Cloudflare Workers
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;

pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying Worker Catalog to gwc.rbee.dev");
    println!();

    let worker_dir = "bin/80-hono-worker-catalog";

    // TEAM-452: Check if wrangler.jsonc exists (not .toml)
    let wrangler_path = format!("{}/wrangler.jsonc", worker_dir);
    if !std::path::Path::new(&wrangler_path).exists() {
        println!("⚠️  wrangler.jsonc not found, creating it...");
        create_wrangler_config(worker_dir, dry_run)?;
    }

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  cd {} && pnpm deploy", worker_dir);
        return Ok(());
    }

    // Deploy
    println!("📦 Deploying worker...");
    let status = Command::new("pnpm")
        .args(&["deploy"])
        .current_dir(worker_dir)
        .status()?;

    if !status.success() {
        anyhow::bail!("Worker deployment failed");
    }

    println!();
    println!("✅ Worker catalog deployed!");
    println!("🌐 URL: https://gwc.rbee.dev");
    println!();
    println!("Verify:");
    println!("  curl https://gwc.rbee.dev/health");

    Ok(())
}

fn create_wrangler_config(worker_dir: &str, dry_run: bool) -> Result<()> {
    let content = r#"{
  "name": "rbee-worker-catalog",
  "main": "src/index.ts",
  "compatibility_date": "2024-11-01",
  "env": {
    "production": {
      "name": "rbee-worker-catalog",
      "routes": [
        { "pattern": "gwc.rbee.dev", "custom_domain": true }
      ]
    }
  }
}
"#;

    if dry_run {
        println!("Would create wrangler.jsonc:");
        println!("{}", content);
        return Ok(());
    }

    let path = format!("{}/wrangler.jsonc", worker_dir);
    std::fs::write(&path, content)?;
    println!("✅ Created {}", path);

    Ok(())
}
