// Bump JavaScript package versions
// Created by: TEAM-451

use anyhow::Result;
use semver::Version;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

use super::tiers::{BumpType, TierConfig};

pub fn bump_js_packages(
    config: &TierConfig,
    bump_type: &BumpType,
    dry_run: bool,
) -> Result<Vec<(String, Version, Version)>> {
    let mut changes = Vec::new();

    for package_name in &config.javascript.packages {
        match find_and_bump_package(package_name, bump_type, dry_run) {
            Ok((old, new)) => {
                changes.push((package_name.clone(), old, new));
            }
            Err(e) => {
                eprintln!("Warning: Failed to bump {}: {}", package_name, e);
            }
        }
    }

    Ok(changes)
}

fn find_and_bump_package(
    package_name: &str,
    bump_type: &BumpType,
    dry_run: bool,
) -> Result<(Version, Version)> {
    // Search for package.json with matching name
    let package_json_path = find_package_json(package_name)?;

    // Read and parse
    let content = fs::read_to_string(&package_json_path)?;
    let mut json: Value = serde_json::from_str(&content)?;

    // Get current version
    let current_version = json["version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No version found in {}", package_json_path.display()))?;

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

    // Update JSON
    json["version"] = Value::String(version.to_string());

    // Write back (unless dry-run)
    if !dry_run {
        let pretty = serde_json::to_string_pretty(&json)?;
        fs::write(&package_json_path, format!("{}\n", pretty))?;  // TEAM-452: Add newline
    }

    Ok((old_version, version))
}

fn find_package_json(package_name: &str) -> Result<PathBuf> {
    // Search in common locations
    let search_paths = vec!["frontend", "bin"];

    for search_path in search_paths {
        if let Ok(path) = search_in_directory(search_path, package_name) {
            return Ok(path);
        }
    }

    anyhow::bail!("Could not find package.json for {}", package_name)
}

fn search_in_directory(dir: &str, package_name: &str) -> Result<PathBuf> {
    for entry in WalkDir::new(dir)
        .max_depth(10)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "package.json" {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                    if let Some(name) = json["name"].as_str() {
                        if name == package_name {
                            return Ok(entry.path().to_path_buf());
                        }
                    }
                }
            }
        }
    }

    anyhow::bail!("Package {} not found in {}", package_name, dir)
}
