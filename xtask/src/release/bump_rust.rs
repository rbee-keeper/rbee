// Bump Rust crate versions
// Created by: TEAM-451

use anyhow::Result;
use semver::Version;
use std::fs;
use std::path::PathBuf;
use toml_edit::{value, DocumentMut};

use super::tiers::{get_all_rust_crates, BumpType, TierConfig};

pub fn bump_rust_crates(
    config: &TierConfig,
    bump_type: &BumpType,
    dry_run: bool,
) -> Result<Vec<(PathBuf, Version, Version)>> {
    let mut changes = Vec::new();
    let crates = get_all_rust_crates(config);

    for crate_path in crates {
        let cargo_toml_path = crate_path.join("Cargo.toml");

        if !cargo_toml_path.exists() {
            eprintln!(
                "Warning: Cargo.toml not found at {}",
                cargo_toml_path.display()
            );
            continue;
        }

        match bump_cargo_toml(&cargo_toml_path, bump_type, dry_run) {
            Ok((old, new)) => {
                changes.push((crate_path, old, new));
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to bump {}: {}",
                    cargo_toml_path.display(),
                    e
                );
            }
        }
    }

    Ok(changes)
}

fn bump_cargo_toml(
    path: &PathBuf,
    bump_type: &BumpType,
    dry_run: bool,
) -> Result<(Version, Version)> {
    // Read and parse
    let content = fs::read_to_string(path)?;
    let mut doc = content.parse::<DocumentMut>()?;

    // Get current version
    let current_version = doc["package"]["version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No version found in {}", path.display()))?;

    let mut version = Version::parse(current_version)?;
    let old_version = version.clone();

    // Bump version
    match bump_type {
        BumpType::Patch => {
            version.patch += 1;
        }
        BumpType::Minor => {
            version.minor += 1;
            version.patch = 0;
        }
        BumpType::Major => {
            version.major += 1;
            version.minor = 0;
            version.patch = 0;
        }
    }

    // Update document
    doc["package"]["version"] = value(version.to_string());

    // Write back (unless dry-run)
    if !dry_run {
        fs::write(path, doc.to_string())?;
    }

    Ok((old_version, version))
}
