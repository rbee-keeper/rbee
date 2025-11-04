# URL Protocol Registration Guide

**Date:** 2025-11-04  
**Purpose:** Step-by-step guide to register `rbee://` protocol on all platforms

---

## 🐧 Linux

### Method 1: .desktop File (Recommended)

**1. Create desktop entry file:**

```bash
# File: ~/.local/share/applications/rbee-keeper.desktop
[Desktop Entry]
Version=1.0
Name=rbee Keeper
Comment=rbee Keeper - AI Infrastructure Manager
Exec=/usr/local/bin/rbee-keeper %u
Icon=/usr/share/icons/rbee-keeper.png
Terminal=false
Type=Application
Categories=Development;Utility;
MimeType=x-scheme-handler/rbee;
```

**2. Update desktop database:**

```bash
update-desktop-database ~/.local/share/applications/
```

**3. Register as default handler:**

```bash
xdg-mime default rbee-keeper.desktop x-scheme-handler/rbee
```

**4. Test:**

```bash
xdg-open "rbee://download/model/huggingface/llama-3.2-1b"
```

### Method 2: During Installation (System-wide)

**File: `bin/00_rbee_keeper/install.sh`**

```bash
#!/bin/bash

# Install binary
sudo cp target/release/rbee-keeper /usr/local/bin/

# Install icon
sudo cp assets/icon.png /usr/share/icons/rbee-keeper.png

# Install .desktop file
cat > /tmp/rbee-keeper.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=rbee Keeper
Comment=rbee Keeper - AI Infrastructure Manager
Exec=/usr/local/bin/rbee-keeper %u
Icon=/usr/share/icons/rbee-keeper.png
Terminal=false
Type=Application
Categories=Development;Utility;
MimeType=x-scheme-handler/rbee;
EOF

sudo cp /tmp/rbee-keeper.desktop /usr/share/applications/
sudo update-desktop-database /usr/share/applications/

# Register protocol handler
xdg-mime default rbee-keeper.desktop x-scheme-handler/rbee

echo "✅ rbee:// protocol registered"
echo "Test with: xdg-open 'rbee://test'"
```

### Method 3: Programmatically (Rust)

**Crate:** `open` or custom implementation

```rust
// bin/00_rbee_keeper/src/protocol_registration.rs

use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub fn register_protocol() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    {
        register_linux()?;
    }
    
    Ok(())
}

#[cfg(target_os = "linux")]
fn register_linux() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let desktop_dir = PathBuf::from(&home)
        .join(".local/share/applications");
    
    // Create directory if it doesn't exist
    fs::create_dir_all(&desktop_dir)?;
    
    // Get current binary path
    let exe_path = std::env::current_exe()?;
    
    // Create .desktop file
    let desktop_content = format!(
        r#"[Desktop Entry]
Version=1.0
Name=rbee Keeper
Comment=rbee Keeper - AI Infrastructure Manager
Exec={} %u
Terminal=false
Type=Application
Categories=Development;Utility;
MimeType=x-scheme-handler/rbee;
"#,
        exe_path.display()
    );
    
    let desktop_file = desktop_dir.join("rbee-keeper.desktop");
    fs::write(&desktop_file, desktop_content)?;
    
    // Update desktop database
    Command::new("update-desktop-database")
        .arg(&desktop_dir)
        .status()?;
    
    // Register as default handler
    Command::new("xdg-mime")
        .args(&["default", "rbee-keeper.desktop", "x-scheme-handler/rbee"])
        .status()?;
    
    println!("✅ rbee:// protocol registered on Linux");
    Ok(())
}
```

---

## 🍎 macOS

### Method 1: Info.plist (Recommended)

**File: `bin/00_rbee_keeper/Info.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>rbee Keeper</string>
    
    <key>CFBundleIdentifier</key>
    <string>dev.rbee.keeper</string>
    
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    
    <key>CFBundleExecutable</key>
    <string>rbee-keeper</string>
    
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    
    <!-- URL Scheme Registration -->
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleURLName</key>
            <string>rbee Protocol</string>
            
            <key>CFBundleURLSchemes</key>
            <array>
                <string>rbee</string>
            </array>
            
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
        </dict>
    </array>
</dict>
</plist>
```

### Method 2: Create .app Bundle

**Structure:**
```
rbee-keeper.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── rbee-keeper (binary)
│   └── Resources/
│       └── AppIcon.icns
```

**Build script:**

```bash
#!/bin/bash
# File: bin/00_rbee_keeper/build-macos.sh

APP_NAME="rbee-keeper"
APP_DIR="$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Create directories
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy binary
cp target/release/rbee-keeper "$MACOS_DIR/"

# Copy Info.plist
cp Info.plist "$CONTENTS_DIR/"

# Copy icon (if exists)
if [ -f "assets/AppIcon.icns" ]; then
    cp assets/AppIcon.icns "$RESOURCES_DIR/"
fi

echo "✅ macOS app bundle created: $APP_DIR"
echo "Install with: cp -r $APP_DIR /Applications/"
```

### Method 3: Programmatically (Rust)

```rust
#[cfg(target_os = "macos")]
fn register_macos() -> Result<(), Box<dyn std::error::Error>> {
    // On macOS, protocol registration happens via Info.plist
    // This function just verifies it's registered
    
    use std::process::Command;
    
    // Check if protocol is registered
    let output = Command::new("open")
        .args(&["-Ra", "rbee://test"])
        .output()?;
    
    if output.status.success() {
        println!("✅ rbee:// protocol is registered");
    } else {
        println!("⚠️  rbee:// protocol not registered");
        println!("Please install the .app bundle to /Applications/");
    }
    
    Ok(())
}
```

### Method 4: Handle URL Events

**File: `bin/00_rbee_keeper/src/macos_url_handler.rs`**

```rust
// macOS-specific URL event handling
#[cfg(target_os = "macos")]
pub fn setup_url_handler() {
    use cocoa::appkit::{NSApp, NSApplication};
    use cocoa::base::nil;
    use objc::{class, msg_send, sel, sel_impl};
    
    unsafe {
        let app = NSApp();
        let delegate = create_app_delegate();
        app.setDelegate_(delegate);
    }
}

// Implement NSApplicationDelegate to receive URL events
// This is complex - usually use a crate like `fruitbasket`
```

**Better: Use `tauri` or `wry` which handles this automatically**

---

## 🪟 Windows

### Method 1: Registry (Recommended)

**File: `bin/00_rbee_keeper/register-protocol.reg`**

```reg
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\rbee]
@="URL:rbee Protocol"
"URL Protocol"=""

[HKEY_CLASSES_ROOT\rbee\DefaultIcon]
@="C:\\Program Files\\rbee\\keeper.exe,1"

[HKEY_CLASSES_ROOT\rbee\shell]

[HKEY_CLASSES_ROOT\rbee\shell\open]

[HKEY_CLASSES_ROOT\rbee\shell\open\command]
@="\"C:\\Program Files\\rbee\\keeper.exe\" \"%1\""
```

**Import:**
```cmd
regedit /s register-protocol.reg
```

### Method 2: Programmatically (Rust)

**Crate:** `winreg`

```rust
// Cargo.toml
[target.'cfg(windows)'.dependencies]
winreg = "0.52"
```

```rust
// bin/00_rbee_keeper/src/protocol_registration.rs

#[cfg(target_os = "windows")]
fn register_windows() -> Result<(), Box<dyn std::error::Error>> {
    use winreg::enums::*;
    use winreg::RegKey;
    
    let hkcr = RegKey::predef(HKEY_CLASSES_ROOT);
    
    // Get current executable path
    let exe_path = std::env::current_exe()?;
    let exe_str = exe_path.to_str().unwrap();
    
    // Create rbee key
    let (key, _) = hkcr.create_subkey("rbee")?;
    key.set_value("", &"URL:rbee Protocol")?;
    key.set_value("URL Protocol", &"")?;
    
    // Create DefaultIcon key
    let (icon_key, _) = key.create_subkey("DefaultIcon")?;
    icon_key.set_value("", &format!("{},1", exe_str))?;
    
    // Create shell\open\command key
    let (cmd_key, _) = key.create_subkey("shell\\open\\command")?;
    cmd_key.set_value("", &format!("\"{}\" \"%1\"", exe_str))?;
    
    println!("✅ rbee:// protocol registered on Windows");
    Ok(())
}
```

### Method 3: Installer (NSIS)

**File: `bin/00_rbee_keeper/installer.nsi`**

```nsis
; NSIS Installer Script

!define APP_NAME "rbee Keeper"
!define PROTOCOL "rbee"

Section "Install"
    ; Install binary
    SetOutPath "$PROGRAMFILES\rbee"
    File "target\release\rbee-keeper.exe"
    
    ; Register protocol
    WriteRegStr HKCR "${PROTOCOL}" "" "URL:${PROTOCOL} Protocol"
    WriteRegStr HKCR "${PROTOCOL}" "URL Protocol" ""
    WriteRegStr HKCR "${PROTOCOL}\DefaultIcon" "" "$PROGRAMFILES\rbee\keeper.exe,1"
    WriteRegStr HKCR "${PROTOCOL}\shell\open\command" "" '"$PROGRAMFILES\rbee\keeper.exe" "%1"'
    
    ; Create uninstaller
    WriteUninstaller "$PROGRAMFILES\rbee\uninstall.exe"
SectionEnd

Section "Uninstall"
    ; Remove protocol registration
    DeleteRegKey HKCR "${PROTOCOL}"
    
    ; Remove files
    Delete "$PROGRAMFILES\rbee\keeper.exe"
    Delete "$PROGRAMFILES\rbee\uninstall.exe"
    RMDir "$PROGRAMFILES\rbee"
SectionEnd
```

---

## 🦀 Rust Implementation (Cross-Platform)

### Using `tauri` (Recommended)

**Tauri handles protocol registration automatically!**

```toml
# Cargo.toml
[dependencies]
tauri = { version = "1.5", features = ["protocol-all"] }
```

```rust
// src-tauri/tauri.conf.json
{
  "tauri": {
    "bundle": {
      "identifier": "dev.rbee.keeper",
      "protocols": [
        {
          "name": "rbee",
          "schemes": ["rbee"]
        }
      ]
    }
  }
}
```

```rust
// src-tauri/src/main.rs
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Handle protocol URLs
            app.listen_global("rbee-protocol", |event| {
                if let Some(url) = event.payload() {
                    println!("Received URL: {}", url);
                    handle_url_scheme(url);
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn handle_url_scheme(url: &str) {
    // Parse and handle URL
    println!("Handling: {}", url);
}
```

### Manual Implementation (All Platforms)

**File: `bin/00_rbee_keeper/src/protocol_registration.rs`**

```rust
use std::error::Error;

pub fn register_protocol() -> Result<(), Box<dyn Error>> {
    #[cfg(target_os = "linux")]
    register_linux()?;
    
    #[cfg(target_os = "macos")]
    register_macos()?;
    
    #[cfg(target_os = "windows")]
    register_windows()?;
    
    Ok(())
}

#[cfg(target_os = "linux")]
fn register_linux() -> Result<(), Box<dyn Error>> {
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    
    let home = std::env::var("HOME")?;
    let desktop_dir = PathBuf::from(&home).join(".local/share/applications");
    fs::create_dir_all(&desktop_dir)?;
    
    let exe_path = std::env::current_exe()?;
    let desktop_content = format!(
        "[Desktop Entry]\n\
         Version=1.0\n\
         Name=rbee Keeper\n\
         Exec={} %u\n\
         Terminal=false\n\
         Type=Application\n\
         MimeType=x-scheme-handler/rbee;\n",
        exe_path.display()
    );
    
    let desktop_file = desktop_dir.join("rbee-keeper.desktop");
    fs::write(&desktop_file, desktop_content)?;
    
    Command::new("update-desktop-database")
        .arg(&desktop_dir)
        .status()?;
    
    Command::new("xdg-mime")
        .args(&["default", "rbee-keeper.desktop", "x-scheme-handler/rbee"])
        .status()?;
    
    Ok(())
}

#[cfg(target_os = "macos")]
fn register_macos() -> Result<(), Box<dyn Error>> {
    // On macOS, this is handled by Info.plist in the .app bundle
    println!("Protocol registration on macOS is handled by Info.plist");
    Ok(())
}

#[cfg(target_os = "windows")]
fn register_windows() -> Result<(), Box<dyn Error>> {
    use winreg::enums::*;
    use winreg::RegKey;
    
    let hkcr = RegKey::predef(HKEY_CLASSES_ROOT);
    let exe_path = std::env::current_exe()?;
    let exe_str = exe_path.to_str().unwrap();
    
    let (key, _) = hkcr.create_subkey("rbee")?;
    key.set_value("", &"URL:rbee Protocol")?;
    key.set_value("URL Protocol", &"")?;
    
    let (icon_key, _) = key.create_subkey("DefaultIcon")?;
    icon_key.set_value("", &format!("{},1", exe_str))?;
    
    let (cmd_key, _) = key.create_subkey("shell\\open\\command")?;
    cmd_key.set_value("", &format!("\"{}\" \"%1\"", exe_str))?;
    
    Ok(())
}

// Call this on first run
pub fn ensure_protocol_registered() {
    if !is_protocol_registered() {
        if let Err(e) = register_protocol() {
            eprintln!("Failed to register protocol: {}", e);
        }
    }
}

fn is_protocol_registered() -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::path::PathBuf;
        let home = std::env::var("HOME").unwrap_or_default();
        let desktop_file = PathBuf::from(&home)
            .join(".local/share/applications/rbee-keeper.desktop");
        desktop_file.exists()
    }
    
    #[cfg(target_os = "macos")]
    {
        // Check if .app bundle is installed
        std::path::Path::new("/Applications/rbee-keeper.app").exists()
    }
    
    #[cfg(target_os = "windows")]
    {
        use winreg::enums::*;
        use winreg::RegKey;
        
        let hkcr = RegKey::predef(HKEY_CLASSES_ROOT);
        hkcr.open_subkey("rbee").is_ok()
    }
}
```

---

## 🎯 Handling URL Arguments

### Receive URL in Main

**File: `bin/00_rbee_keeper/src/main.rs`**

```rust
use std::env;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    // Check if launched with URL
    if args.len() > 1 {
        let url = &args[1];
        if url.starts_with("rbee://") {
            handle_url_scheme(url);
            return;
        }
    }
    
    // Normal startup
    start_gui();
}

fn handle_url_scheme(url: &str) {
    println!("Received URL: {}", url);
    
    // Parse URL
    if let Ok(parsed) = url::Url::parse(url) {
        let action = parsed.host_str().unwrap_or("");
        let path = parsed.path();
        
        match action {
            "download" => handle_download(path),
            "install" => handle_install(path),
            "open" => handle_open(path),
            _ => eprintln!("Unknown action: {}", action),
        }
    }
}

fn handle_download(path: &str) {
    // path = "/model/huggingface/llama-3.2-1b"
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    
    if parts.len() >= 3 && parts[0] == "model" {
        let source = parts[1];
        let model_id = parts[2];
        
        println!("Download model: {} from {}", model_id, source);
        
        // Show GUI with download dialog
        show_download_dialog(source, model_id);
    }
}
```

### Single Instance (Prevent Multiple Windows)

**Crate:** `single-instance`

```toml
[dependencies]
single-instance = "0.3"
```

```rust
use single_instance::SingleInstance;

fn main() {
    let instance = SingleInstance::new("rbee-keeper").unwrap();
    
    if !instance.is_single() {
        // Another instance is running
        // Send URL to existing instance via IPC
        send_to_existing_instance(&std::env::args().collect::<Vec<_>>());
        return;
    }
    
    // This is the first instance
    start_gui();
}
```

---

## 🧪 Testing

### Test on Linux

```bash
# Test protocol handler
xdg-open "rbee://download/model/huggingface/llama-3.2-1b"

# Check if registered
xdg-mime query default x-scheme-handler/rbee
# Should output: rbee-keeper.desktop
```

### Test on macOS

```bash
# Test protocol handler
open "rbee://download/model/huggingface/llama-3.2-1b"

# Check if registered
open -Ra "rbee://test"
# Should launch rbee-keeper.app
```

### Test on Windows

```cmd
# Test protocol handler
start rbee://download/model/huggingface/llama-3.2-1b

# Check registry
reg query HKCR\rbee
```

### Test from Browser

**HTML file:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>rbee Protocol Test</title>
</head>
<body>
    <h1>rbee Protocol Test</h1>
    
    <a href="rbee://download/model/huggingface/llama-3.2-1b">
        Download Llama 3.2 1B
    </a>
    
    <br><br>
    
    <a href="rbee://install/worker/llm-worker-rbee-cuda">
        Install CUDA Worker
    </a>
    
    <br><br>
    
    <a href="rbee://open/hive/localhost">
        Open Hive
    </a>
</body>
</html>
```

---

## 📦 Recommended Approach

### Option 1: Use Tauri (Easiest)

**Pros:**
- ✅ Handles protocol registration automatically
- ✅ Cross-platform (Linux, macOS, Windows)
- ✅ Built-in IPC for single instance
- ✅ Modern web-based GUI

**Cons:**
- ❌ Requires rewriting UI in Tauri
- ❌ Larger binary size

### Option 2: Manual Registration (Current Setup)

**Pros:**
- ✅ Works with existing Rust + React setup
- ✅ Full control over registration
- ✅ Smaller binary

**Cons:**
- ❌ More code to maintain
- ❌ Platform-specific code

### Recommendation: **Option 2** (Manual Registration)

Since you already have a working Rust + React setup, just add protocol registration to the existing binary.

---

## 🚀 Implementation Checklist

- [ ] Add protocol registration code to `bin/00_rbee_keeper/src/main.rs`
- [ ] Register protocol on first run
- [ ] Handle URL arguments in main()
- [ ] Parse `rbee://` URLs
- [ ] Implement download/install/open handlers
- [ ] Add single-instance check
- [ ] Test on Linux
- [ ] Test on macOS
- [ ] Test on Windows
- [ ] Create installer scripts
- [ ] Update documentation

---

**Start with Linux (easiest), then macOS, then Windows!** 🚀
