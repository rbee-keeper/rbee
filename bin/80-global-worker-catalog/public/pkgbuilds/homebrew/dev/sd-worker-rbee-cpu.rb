# Homebrew Formula for sd-worker-rbee (CPU variant)
# Platform: macOS
# Build: Development (HEAD)

class SdWorkerRbeeCpu < Formula
  desc "Stable Diffusion worker for rbee system (CPU-only, development)"
  homepage "https://rbee.dev"
  url "https://github.com/rbee-keeper/rbee.git", branch: "development"
  license "GPL-3.0-or-later"

  depends_on "rust" => :build

  def install
    cd "bin/31_sd_worker_rbee" do
      system "cargo", "build", "--release", "--no-default-features", "--features", "cpu"
    end
    bin.install "target/release/sd-worker-rbee" => "sd-worker-rbee-cpu"
  end

  test do
    system "#{bin}/sd-worker-rbee-cpu", "--version"
  end
end
