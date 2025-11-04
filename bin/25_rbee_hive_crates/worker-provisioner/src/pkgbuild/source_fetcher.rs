//! Source fetcher for PKGBUILD installation
//!
//! TEAM-378: Handles downloading and extracting sources from PKGBUILD source=() field
//! Supports git repositories with tag/branch/commit syntax

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::path::Path;
use tokio::process::Command;

/// Parse and fetch sources from PKGBUILD source array
///
/// Supports:
/// - `git+https://github.com/user/repo.git#tag=v1.0.0`
/// - `git+https://github.com/user/repo.git#branch=main`
/// - `git+https://github.com/user/repo.git#commit=abc123`
/// - Plain URLs (future: tarballs, etc.)
///
/// TEAM-378: Automatically converts HTTPS GitHub URLs to SSH (git@github.com:user/repo.git)
pub async fn fetch_sources(sources: &[String], srcdir: &Path) -> Result<()> {
    if sources.is_empty() {
        anyhow::bail!("No sources specified in PKGBUILD. Cannot proceed.");
    }

    for (idx, source) in sources.iter().enumerate() {
        n!("fetch_source", "📥 Fetching source {}/{}: {}", idx + 1, sources.len(), source);
        
        if source.starts_with("git+") {
            fetch_git_source(source, srcdir).await?;
        } else if source.starts_with("http://") || source.starts_with("https://") {
            // TODO: Implement tarball/file download
            anyhow::bail!("HTTP/HTTPS file downloads not yet implemented. Use git+ URLs for now.");
        } else {
            n!("fetch_source_skip", "⏭️  Skipping unknown source type: {}", source);
        }
    }

    Ok(())
}

/// Fetch a git repository
///
/// Syntax: `git+https://github.com/user/repo.git#tag=v1.0.0`
/// Syntax: `git+https://github.com/user/repo.git#branch=main`
/// Syntax: `git+https://github.com/user/repo.git#commit=abc123`
///
/// TEAM-378: Converts HTTPS GitHub URLs to SSH format automatically
async fn fetch_git_source(source: &str, srcdir: &Path) -> Result<()> {
    // Parse git URL
    let source = source.strip_prefix("git+").unwrap_or(source);
    
    let (url, ref_spec) = if let Some(pos) = source.find('#') {
        let (url_part, ref_part) = source.split_at(pos);
        let ref_part = &ref_part[1..]; // Skip the '#'
        (url_part, Some(ref_part))
    } else {
        (source, None)
    };

    // TEAM-378: Convert HTTPS GitHub URLs to SSH
    let url = convert_https_to_ssh(url);
    
    n!("git_clone_start", "🔄 Cloning repository: {}", url);
    
    // Extract repo name from URL for directory name
    let repo_name = url
        .rsplit('/')
        .next()
        .unwrap_or("repo")
        .trim_end_matches(".git");
    
    let clone_dir = srcdir.join(repo_name);

    // TEAM-378: Clean up existing directory if it exists (for dev retries)
    if clone_dir.exists() {
        n!("git_cleanup", "🧹 Removing existing directory: {}", clone_dir.display());
        tokio::fs::remove_dir_all(&clone_dir)
            .await
            .context("Failed to remove existing clone directory")?;
    }

    // Clone the repository
    let mut cmd = Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1"); // Shallow clone for speed

    // Add branch/tag if specified
    if let Some(ref_spec) = ref_spec {
        if let Some(tag) = ref_spec.strip_prefix("tag=") {
            n!("git_clone_tag", "🏷️  Checking out tag: {}", tag);
            cmd.arg("--branch").arg(tag);
        } else if let Some(branch) = ref_spec.strip_prefix("branch=") {
            n!("git_clone_branch", "🌿 Checking out branch: {}", branch);
            cmd.arg("--branch").arg(branch);
        } else if let Some(commit) = ref_spec.strip_prefix("commit=") {
            // For commits, we need to clone full repo then checkout
            n!("git_clone_commit", "📌 Will checkout commit: {}", commit);
            // Remove --depth for commit checkout
            cmd = Command::new("git");
            cmd.arg("clone");
        }
    }

    cmd.arg(&url).arg(&clone_dir);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    n!("git_clone_exec", "⚙️  Executing: git clone {} -> {}", url, clone_dir.display());
    
    // TEAM-378: Spawn process and stream output in real-time
    let mut child = cmd.spawn().context("Failed to spawn git clone")?;
    
    // Get stdout and stderr handles
    let stdout = child.stdout.take().context("Failed to get stdout")?;
    let stderr = child.stderr.take().context("Failed to get stderr")?;
    
    // Stream stdout
    let stdout_task = tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            n!("git_stdout", "  {}", line);
        }
    });
    
    // Stream stderr (git progress goes to stderr)
    let stderr_task = tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            n!("git_stderr", "  {}", line);
        }
    });
    
    // Wait for process with timeout
    let status = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        child.wait()
    )
    .await
    .context("Git clone timed out after 60 seconds")?
    .context("Failed to wait for git clone")?;
    
    // Wait for output tasks to complete
    let _ = tokio::join!(stdout_task, stderr_task);

    if !status.success() {
        anyhow::bail!("Git clone failed (exit code: {:?})", status.code());
    }

    n!("git_clone_ok", "✓ Repository cloned to: {}", clone_dir.display());

    // If commit was specified, checkout the specific commit
    if let Some(ref_spec) = ref_spec {
        if let Some(commit) = ref_spec.strip_prefix("commit=") {
            n!("git_checkout_commit", "📌 Checking out commit: {}", commit);
            
            let output = Command::new("git")
                .arg("-C")
                .arg(&clone_dir)
                .arg("checkout")
                .arg(commit)
                .output()
                .await
                .context("Failed to checkout commit")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("Git checkout failed: {}", stderr);
            }

            n!("git_checkout_ok", "✓ Checked out commit: {}", commit);
        }
    }

    Ok(())
}

/// Convert HTTPS GitHub URLs to SSH format
///
/// TEAM-378: Converts https://github.com/user/repo.git to git@github.com:user/repo.git
fn convert_https_to_ssh(url: &str) -> String {
    // Check if it's a GitHub HTTPS URL
    if url.starts_with("https://github.com/") {
        // Extract the path after github.com/
        let path = url.strip_prefix("https://github.com/").unwrap();
        let ssh_url = format!("git@github.com:{}", path);
        n!("url_convert", "🔄 Converting HTTPS to SSH: {} -> {}", url, ssh_url);
        ssh_url
    } else {
        // Return as-is for non-GitHub URLs or already SSH URLs
        url.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_https_to_ssh() {
        let https_url = "https://github.com/veighnsche/llama-orch.git";
        let ssh_url = convert_https_to_ssh(https_url);
        assert_eq!(ssh_url, "git@github.com:veighnsche/llama-orch.git");
    }

    #[test]
    fn test_convert_non_github_url() {
        let url = "https://gitlab.com/user/repo.git";
        let result = convert_https_to_ssh(url);
        assert_eq!(result, url); // Should return unchanged
    }

    #[test]
    fn test_convert_already_ssh() {
        let ssh_url = "git@github.com:user/repo.git";
        let result = convert_https_to_ssh(ssh_url);
        assert_eq!(result, ssh_url); // Should return unchanged
    }
}
