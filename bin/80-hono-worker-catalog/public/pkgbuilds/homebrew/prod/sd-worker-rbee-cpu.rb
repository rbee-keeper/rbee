# Homebrew Formula for sd-worker-rbee (CPU variant)
# Platform: macOS
# Build: Production (bottles)

class SdWorkerRbeeCpu < Formula
  desc "Stable Diffusion worker for rbee system (CPU-only)"
  homepage "https://rbee.dev"
  url "https://github.com/rbee-keeper/rbee/releases/download/v0.1.0/sd-worker-rbee-macos-arm64-0.1.0.tar.gz"
  sha256 "SKIP"
  license "GPL-3.0-or-later"

  def install
    bin.install "sd-worker-rbee" => "sd-worker-rbee-cpu"
  end

  test do
    system "#{bin}/sd-worker-rbee-cpu", "--version"
  end
end
