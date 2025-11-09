// Admin Worker Deployment
// Cloudflare Worker for admin dashboard, analytics, and user management
// Location: bin/78-admin

use anyhow::{Context, Result};
use colored::Colorize;
use std::process::Command;

pub fn deploy(production: bool, dry_run: bool) -> Result<()> {
    let app_name = "Admin Worker";
    let domain = if production {
        "install.rbee.dev"
    } else {
        "install-dev.rbee.dev"
    };
    
    println!("{}", format!("🚀 Deploying {} to {}", app_name, domain).bright_cyan().bold());
    println!();

    // Change to admin directory
    let admin_dir = "bin/78-admin";
    
    if dry_run {
        println!("🔍 Dry run - would deploy from: {}", admin_dir);
        println!("  Domain: {}", domain);
        println!("  Production: {}", production);
        return Ok(());
    }

    // Step 1: Build Tailwind CSS
    println!("🎨 Building Tailwind CSS...");
    let build_css = Command::new("npm")
        .args(["run", "build:css"])
        .current_dir(admin_dir)
        .status()
        .context("Failed to build Tailwind CSS")?;

    if !build_css.success() {
        anyhow::bail!("Tailwind CSS build failed");
    }
    println!("  ✅ Tailwind CSS built");
    println!();

    // Step 2: Run tests
    println!("🧪 Running tests...");
    let test = Command::new("npm")
        .args(["test"])
        .current_dir(admin_dir)
        .status()
        .context("Failed to run tests")?;

    if !test.success() {
        anyhow::bail!("Tests failed - fix tests before deploying");
    }
    println!("  ✅ All tests passed");
    println!();

    // Step 3: Deploy to Cloudflare
    println!("☁️  Deploying to Cloudflare Workers...");
    
    let mut deploy_cmd = Command::new("wrangler");
    deploy_cmd
        .args(["deploy", "--minify"])
        .current_dir(admin_dir);
    
    if production {
        deploy_cmd.env("ENVIRONMENT", "production");
    }
    
    let deploy = deploy_cmd
        .status()
        .context("Failed to deploy to Cloudflare")?;

    if !deploy.success() {
        anyhow::bail!("Cloudflare deployment failed");
    }

    println!();
    println!("{}", "✅ Deployment successful!".bright_green().bold());
    println!();
    println!("📍 Deployed to: https://{}", domain);
    println!();
    
    // Show important endpoints
    println!("{}", "🔗 Important Endpoints:".bright_cyan());
    println!("  Admin Dashboard:  https://{}/admin", domain);
    println!("  User Dashboard:   https://{}/dashboard", domain);
    println!("  Analytics SDK:    https://{}/analytics.js", domain);
    println!("  Health Check:     https://{}/health", domain);
    println!();
    
    // Show next steps
    println!("{}", "📋 Post-Deployment Checklist:".bright_yellow());
    println!("  1. Verify health check: curl https://{}/health", domain);
    println!("  2. Test admin dashboard (with token)");
    println!("  3. Test analytics tracking");
    println!("  4. Check Cloudflare dashboard for errors");
    println!("  5. Monitor logs for issues");
    println!();

    Ok(())
}
