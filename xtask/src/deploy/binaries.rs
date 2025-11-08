// Deploy Rust binaries to GitHub Releases
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;

pub fn deploy_keeper(dry_run: bool) -> Result<()> {
    deploy_binary("rbee-keeper", "main", dry_run)
}

pub fn deploy_queen(dry_run: bool) -> Result<()> {
    deploy_binary("queen-rbee", "main", dry_run)
}

pub fn deploy_hive(dry_run: bool) -> Result<()> {
    deploy_binary("rbee-hive", "main", dry_run)
}

pub fn deploy_llm_worker(dry_run: bool) -> Result<()> {
    deploy_binary("llm-worker-rbee", "llm-worker", dry_run)
}

pub fn deploy_sd_worker(dry_run: bool) -> Result<()> {
    deploy_binary("sd-worker-rbee", "sd-worker", dry_run)
}

fn deploy_binary(binary_name: &str, tier: &str, dry_run: bool) -> Result<()> {
    println!("🚀 Deploying {} to GitHub Releases", binary_name);
    println!();

    // Get version from tier
    let version = get_tier_version(tier)?;
    let tag = format!("v{}", version);

    println!("📋 Binary: {}", binary_name);
    println!("📦 Tier: {}", tier);
    println!("🏷️  Version: {}", version);
    println!("🔖 Tag: {}", tag);
    println!();

    if dry_run {
        println!("🔍 Dry run - would execute:");
        println!("  1. Build on mac: cargo build --release --package {}", binary_name);
        println!("  2. Package: tar -czf {}-macos-arm64-{}.tar.gz {}", binary_name, version, binary_name);
        println!("  3. Build on blep: cargo build --release --package {}", binary_name);
        println!("  4. Package: tar -czf {}-linux-x86_64-{}.tar.gz {}", binary_name, version, binary_name);
        println!("  5. Create release: gh release create {} --title '{} {}' --notes 'Release notes'", tag, binary_name, version);
        println!("  6. Upload: gh release upload {} {}-macos-arm64-{}.tar.gz {}-linux-x86_64-{}.tar.gz", 
                 tag, binary_name, version, binary_name, version);
        return Ok(());
    }

    // Build on mac
    println!("🔨 Building on mac...");
    build_on_mac(binary_name)?;

    // Package mac binary
    println!("📦 Packaging mac binary...");
    let mac_tarball = package_mac_binary(binary_name, &version)?;

    // Download from mac
    println!("📥 Downloading from mac...");
    download_from_mac(&mac_tarball)?;

    // Build on blep
    println!("🔨 Building on blep...");
    build_on_blep(binary_name)?;

    // Package blep binary
    println!("📦 Packaging blep binary...");
    let linux_tarball = package_blep_binary(binary_name, &version)?;

    // Create or update GitHub release
    println!("🚀 Creating GitHub release...");
    create_or_update_release(&tag, binary_name, &version)?;

    // Upload binaries
    println!("📤 Uploading binaries...");
    upload_to_release(&tag, &mac_tarball, &linux_tarball)?;

    println!();
    println!("✅ {} deployed!", binary_name);
    println!("🌐 Release: https://github.com/rbee-keeper/rbee/releases/tag/{}", tag);
    println!();
    println!("Download:");
    println!("  Mac: gh release download {} --pattern '*macos*'", tag);
    println!("  Linux: gh release download {} --pattern '*linux*'", tag);

    Ok(())
}

fn get_tier_version(_tier: &str) -> Result<String> {
    // All tiers use the workspace version from root Cargo.toml
    let cargo_toml = std::fs::read_to_string("Cargo.toml")?;
    let config: toml::Value = toml::from_str(&cargo_toml)?;
    
    let version = config
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Version not found in Cargo.toml [workspace.package]"))?;
    
    Ok(version.to_string())
}

fn build_on_mac(binary_name: &str) -> Result<()> {
    let status = Command::new("ssh")
        .args(&[
            "mac",
            &format!(
                "cd ~/Projects/rbee && cargo build --release --package {}",
                binary_name
            ),
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Build on mac failed");
    }

    Ok(())
}

fn package_mac_binary(binary_name: &str, version: &str) -> Result<String> {
    let tarball = format!("{}-macos-arm64-{}.tar.gz", binary_name, version);
    
    let status = Command::new("ssh")
        .args(&[
            "mac",
            &format!(
                "cd ~/Projects/rbee/target/release && tar -czf {} {}",
                tarball, binary_name
            ),
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Packaging on mac failed");
    }

    Ok(tarball)
}

fn download_from_mac(tarball: &str) -> Result<()> {
    let status = Command::new("scp")
        .args(&[
            &format!("mac:~/Projects/rbee/target/release/{}", tarball),
            ".",
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Download from mac failed");
    }

    Ok(())
}

fn build_on_blep(binary_name: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["build", "--release", "--package", binary_name])
        .status()?;

    if !status.success() {
        anyhow::bail!("Build on blep failed");
    }

    Ok(())
}

fn package_blep_binary(binary_name: &str, version: &str) -> Result<String> {
    let tarball = format!("{}-linux-x86_64-{}.tar.gz", binary_name, version);
    
    let status = Command::new("tar")
        .args(&[
            "-czf",
            &tarball,
            "-C",
            "target/release",
            binary_name,
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Packaging on blep failed");
    }

    Ok(tarball)
}

fn create_or_update_release(tag: &str, binary_name: &str, version: &str) -> Result<()> {
    // Check if release exists
    let check = Command::new("gh")
        .args(&["release", "view", tag])
        .output()?;

    if check.status.success() {
        println!("ℹ️  Release {} already exists, will upload to it", tag);
        return Ok(());
    }

    // Create new release
    let status = Command::new("gh")
        .args(&[
            "release",
            "create",
            tag,
            "--title",
            &format!("{} {}", binary_name, version),
            "--notes",
            &format!("Release {} version {}", binary_name, version),
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to create release");
    }

    Ok(())
}

fn upload_to_release(tag: &str, mac_tarball: &str, linux_tarball: &str) -> Result<()> {
    let status = Command::new("gh")
        .args(&[
            "release",
            "upload",
            tag,
            mac_tarball,
            linux_tarball,
            "--clobber", // Overwrite if exists
        ])
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to upload binaries");
    }

    Ok(())
}
