# J.A.R.V.I.S. desktop wrapper (Tauri 2 scaffold)

Wraps the web interface as a native macOS app (PRD §3 "Platform Compatibility").

## Prerequisites

- Rust (`rustup`) with the `aarch64-apple-darwin` toolchain
- Tauri CLI: `cargo install tauri-cli --version "^2"`

## Run (development)

The Tauri shell loads `http://localhost:3000`, so boot the stack first:

```bash
# terminal 1 - backend
cd ../../backend && ./run.sh

# terminal 2 - web interface
cd .. && npm run dev

# terminal 3 - native window
cargo tauri dev
```

## Bundle a release app

```bash
cargo tauri icon path/to/icon.png   # generates src-tauri/icons/
cargo tauri build                   # produces a .dmg / .app
```

The release build expects the Next.js server (`node server.mjs`) to be
running locally; for a fully self-contained bundle, spawn it from
`src/main.rs` via `std::process::Command` before the window opens.
