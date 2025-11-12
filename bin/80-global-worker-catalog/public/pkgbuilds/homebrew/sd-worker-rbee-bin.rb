# Homebrew formula for sd-worker-rbee (binary/release version)
# Maintainer: rbee Core Team
# TEAM-481: Downloads pre-built binary from GitHub releases
# Auto-detects platform: Metal (Apple Silicon) or CPU (Intel)

class SdWorkerRbeeBin < Formula
  desc "Stable Diffusion worker for rbee system (pre-built binary)"
  homepage "https://github.com/rbee-keeper/rbee"
  version "0.1.0"
  license "GPL-3.0-or-later"

  # Auto-detect platform and best backend
  if Hardware::CPU.arm?
    # Apple Silicon: Metal acceleration
    url "https://github.com/rbee-keeper/rbee/releases/download/v#{version}/sd-worker-rbee-macos-aarch64-metal-#{version}.tar.gz"
    sha256 "SKIP"
  else
    # Intel Mac: CPU-only
    url "https://github.com/rbee-keeper/rbee/releases/download/v#{version}/sd-worker-rbee-macos-x86_64-cpu-#{version}.tar.gz"
    sha256 "SKIP"
  end

  def install
    bin.install "sd-worker-rbee"
  end

  test do
    system "#{bin}/sd-worker-rbee", "--version"
  end
end
