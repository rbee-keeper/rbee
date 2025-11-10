# Worker Provisioner

**TEAM-402:** Worker installation from PKGBUILD files

## Overview

This crate handles worker installation for rbee-hive, supporting both source builds and binary packages following the AUR (Arch User Repository) pattern.

## Features

- ✅ **PKGBUILD Parsing** - Parse Arch Linux PKGBUILD format
- ✅ **Source Builds** - Clone git repos and build from source
- ✅ **Binary Packages** - Download and install pre-built binaries
- ✅ **Architecture Support** - `source_x86_64`, `source_aarch64`
- ✅ **Cancellable Operations** - Cancel long-running builds
- ✅ **Progress Tracking** - Real-time build output
- 🚧 **Premium Workers** - Licensed binary distribution (Phase 4)

## Architecture

```
WorkerProvisioner (implements ArtifactProvisioner<WorkerBinary>)
    ↓
CatalogClient → Fetch worker metadata + PKGBUILD
    ↓
PKGBUILD Parser → Parse metadata, functions, sources
    ↓
Source Fetcher → Download sources (git, tarballs)
    ↓
PKGBUILD Executor → Execute build() and package()
    ↓
WorkerBinary installed to ~/.cache/rbee/workers/
```

## Usage

```rust
use rbee_hive_worker_provisioner::WorkerProvisioner;
use rbee_hive_artifact_catalog::ArtifactProvisioner;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create provisioner
    let provisioner = WorkerProvisioner::new()?;
    
    // Provision a worker
    let cancel_token = CancellationToken::new();
    let worker = provisioner.provision(
        "llm-worker-rbee-cpu",
        "job-123",
        cancel_token
    ).await?;
    
    println!("Worker installed: {:?}", worker);
    Ok(())
}
```

## PKGBUILD Support

### Source Build (Standard)

```bash
pkgname=llm-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
source=("git+https://github.com/rbee-keeper/rbee.git#branch=main")

build() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo build --release --features cpu
}

package() {
    install -Dm755 "target/release/llm-worker-rbee-cpu" \
        "$pkgdir/usr/local/bin/$pkgname"
}
```

### Binary Package (AUR Pattern)

```bash
pkgname=llm-worker-rbee-premium
pkgver=0.1.0
pkgrel=1
source_x86_64=("https://releases.rbee.ai/premium-x86_64.tar.gz")
source_aarch64=("https://releases.rbee.ai/premium-aarch64.tar.gz")
sha256sums_x86_64=('abc123...')
sha256sums_aarch64=('def456...')

# NO build() function for binary packages!

package() {
    install -Dm755 "llm-worker-rbee-premium" \
        "$pkgdir/usr/local/bin/$pkgname"
}
```

## Migration from rbee-hive

This crate was created by migrating code from `bin/20_rbee_hive/src/`:

- `pkgbuild_parser.rs` → `src/pkgbuild/parser.rs`
- `pkgbuild_executor.rs` → `src/pkgbuild/executor.rs`
- `source_fetcher.rs` → `src/pkgbuild/source_fetcher.rs`
- `worker_install.rs` → `src/provisioner.rs` (refactored)

See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for details.

## Testing

```bash
cargo test
```

All tests from the original rbee-hive implementation are preserved.

## Documentation

- [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) - Migration from rbee-hive
- [AUR_ENHANCEMENTS.md](./AUR_ENHANCEMENTS.md) - New AUR features (Phase 4)

---

**TEAM-402 - Worker Provisioner**
