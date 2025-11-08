// Deployment gates - tests that must pass before deployment
// Created by: TEAM-451

use anyhow::Result;
use std::process::Command;

/// Run all deployment gates for an app
pub fn check_gates(app: &str) -> Result<()> {
    println!("🚦 Running deployment gates for {}...", app);
    println!();

    match app {
        // Cloudflare apps
        "worker" | "gwc" | "worker-catalog" => check_worker_catalog_gates()?,
        "commercial" => check_commercial_gates()?,
        "marketplace" => check_marketplace_gates()?,
        "docs" | "user-docs" => check_docs_gates()?,
        
        // Rust binaries
        "keeper" | "rbee-keeper" => check_keeper_gates()?,
        "queen" | "queen-rbee" => check_queen_gates()?,
        "hive" | "rbee-hive" => check_hive_gates()?,
        "llm-worker" | "llm-worker-rbee" => check_llm_worker_gates()?,
        "sd-worker" | "sd-worker-rbee" => check_sd_worker_gates()?,
        
        _ => anyhow::bail!("Unknown app: {}", app),
    }

    println!();
    println!("✅ All deployment gates passed for {}", app);
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLOUDFLARE GATES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn check_worker_catalog_gates() -> Result<()> {
    println!("📦 Worker Catalog Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "bin/80-hono-worker-catalog")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "bin/80-hono-worker-catalog")?;
    
    // Gate 3: Build test
    println!("  3. Build test...");
    run_command("pnpm", &["build"], "bin/80-hono-worker-catalog")?;
    
    // Gate 4: Data validation - verify worker catalog data is valid
    println!("  4. Data validation...");
    validate_worker_catalog_data()?;
    
    // Gate 5: Start dev server and test endpoints
    println!("  5. API endpoint tests...");
    test_worker_catalog_endpoints()?;
    
    Ok(())
}

fn validate_worker_catalog_data() -> Result<()> {
    use std::fs;
    
    // Read and parse data.ts to verify it's valid
    let data_path = "bin/80-hono-worker-catalog/src/data.ts";
    let data_content = fs::read_to_string(data_path)?;
    
    // Check for required worker variants
    let required_workers = vec![
        "llm-worker-rbee-cpu",
        "llm-worker-rbee-cuda",
        "llm-worker-rbee-metal",
        "sd-worker-rbee-cpu",
        "sd-worker-rbee-cuda",
    ];
    
    for worker_id in required_workers {
        if !data_content.contains(worker_id) {
            anyhow::bail!("Missing required worker variant: {}", worker_id);
        }
    }
    
    // Verify all workers have required fields
    let required_fields = vec![
        "id:",
        "implementation:",
        "workerType:",
        "version:",
        "platforms:",
        "architectures:",
        "name:",
        "description:",
        "license:",
    ];
    
    for field in required_fields {
        if !data_content.contains(field) {
            anyhow::bail!("Worker data missing required field: {}", field);
        }
    }
    
    println!("    ✅ All worker variants present");
    println!("    ✅ All required fields present");
    
    Ok(())
}

fn test_worker_catalog_endpoints() -> Result<()> {
    use std::process::{Child, Stdio};
    use std::thread;
    use std::time::Duration;
    
    // Start dev server in background
    println!("    Starting dev server...");
    let mut server = Command::new("pnpm")
        .args(&["dev"])
        .current_dir("bin/80-hono-worker-catalog")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    
    // Wait for server to start
    thread::sleep(Duration::from_secs(3));
    
    // Test endpoints
    let test_result = (|| -> Result<()> {
        // Test 1: Health check
        println!("    Testing /health endpoint...");
        test_endpoint("http://localhost:8787/health", "status")?;
        
        // Test 2: List all workers
        println!("    Testing /workers endpoint...");
        test_endpoint("http://localhost:8787/workers", "workers")?;
        
        // Test 3: Get specific worker
        println!("    Testing /workers/:id endpoint...");
        test_endpoint("http://localhost:8787/workers/llm-worker-rbee-cpu", "id")?;
        
        // Test 4: 404 for invalid worker
        println!("    Testing 404 handling...");
        test_endpoint_404("http://localhost:8787/workers/invalid-worker")?;
        
        println!("    ✅ All endpoints responding correctly");
        
        Ok(())
    })();
    
    // Kill server
    server.kill()?;
    
    test_result
}

fn test_endpoint(url: &str, expected_field: &str) -> Result<()> {
    let output = Command::new("curl")
        .args(&["-s", "-f", url])
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("Endpoint {} returned error", url);
    }
    
    let response = String::from_utf8(output.stdout)?;
    
    if !response.contains(expected_field) {
        anyhow::bail!("Endpoint {} missing expected field: {}", url, expected_field);
    }
    
    Ok(())
}

fn test_endpoint_404(url: &str) -> Result<()> {
    let output = Command::new("curl")
        .args(&["-s", "-w", "%{http_code}", "-o", "/dev/null", url])
        .output()?;
    
    let status_code = String::from_utf8(output.stdout)?;
    
    if status_code != "404" {
        anyhow::bail!("Expected 404 for {}, got {}", url, status_code);
    }
    
    Ok(())
}

fn check_commercial_gates() -> Result<()> {
    println!("🏢 Commercial Site Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/commercial")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/commercial")?;
    
    // Gate 3: Build test
    println!("  3. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/commercial")?;
    
    Ok(())
}

fn check_marketplace_gates() -> Result<()> {
    println!("🛒 Marketplace Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/marketplace")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/marketplace")?;
    
    // Gate 3: Build test
    println!("  3. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/marketplace")?;
    
    Ok(())
}

fn check_docs_gates() -> Result<()> {
    println!("📚 User Docs Gates:");
    
    // Gate 1: TypeScript type check
    println!("  1. TypeScript type check...");
    run_command("pnpm", &["type-check"], "frontend/apps/user-docs")?;
    
    // Gate 2: Lint
    println!("  2. Lint check...");
    run_command("pnpm", &["lint"], "frontend/apps/user-docs")?;
    
    // Gate 3: Build test
    println!("  3. Build test...");
    run_command("pnpm", &["build"], "frontend/apps/user-docs")?;
    
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RUST BINARY GATES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn check_keeper_gates() -> Result<()> {
    println!("🐝 rbee-keeper Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("rbee-keeper")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("rbee-keeper")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("rbee-keeper")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("rbee-keeper")?;
    
    Ok(())
}

fn check_queen_gates() -> Result<()> {
    println!("👑 queen-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("queen-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("queen-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("queen-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("queen-rbee")?;
    
    Ok(())
}

fn check_hive_gates() -> Result<()> {
    println!("🏠 rbee-hive Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("rbee-hive")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("rbee-hive")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("rbee-hive")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("rbee-hive")?;
    
    Ok(())
}

fn check_llm_worker_gates() -> Result<()> {
    println!("🤖 llm-worker-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("llm-worker-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("llm-worker-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("llm-worker-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("llm-worker-rbee")?;
    
    Ok(())
}

fn check_sd_worker_gates() -> Result<()> {
    println!("🎨 sd-worker-rbee Gates:");
    
    // Gate 1: Cargo check
    println!("  1. Cargo check...");
    run_cargo_check("sd-worker-rbee")?;
    
    // Gate 2: Cargo test
    println!("  2. Cargo test...");
    run_cargo_test("sd-worker-rbee")?;
    
    // Gate 3: Cargo clippy
    println!("  3. Cargo clippy...");
    run_cargo_clippy("sd-worker-rbee")?;
    
    // Gate 4: Build test
    println!("  4. Build test...");
    run_cargo_build("sd-worker-rbee")?;
    
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPER FUNCTIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn run_command(cmd: &str, args: &[&str], cwd: &str) -> Result<()> {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .status()?;

    if !status.success() {
        anyhow::bail!("Command failed: {} {} (in {})", cmd, args.join(" "), cwd);
    }

    Ok(())
}

fn run_cargo_check(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["check", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo check failed for {}", package);
    }

    Ok(())
}

fn run_cargo_test(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["test", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo test failed for {}", package);
    }

    Ok(())
}

fn run_cargo_clippy(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["clippy", "--package", package, "--", "-D", "warnings"])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo clippy failed for {}", package);
    }

    Ok(())
}

fn run_cargo_build(package: &str) -> Result<()> {
    let status = Command::new("cargo")
        .args(&["build", "--release", "--package", package])
        .status()?;

    if !status.success() {
        anyhow::bail!("cargo build failed for {}", package);
    }

    Ok(())
}
