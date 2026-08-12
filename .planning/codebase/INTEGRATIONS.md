# External Integrations

**Analysis Date:** 2026-08-09

## APIs & External Services

**Package Registries:**
- crates.io - Rust package publication
  - Package: `anofox-forecast` @ 0.15.8
  - URL: https://crates.io/crates/anofox-forecast
  - Auto-excluded from tarball: validation data (CSV/TSF/JSON/Parquet), generated docs

- npm (npmjs.com) - JavaScript/WebAssembly package publication
  - Package: `@sipemu/anofox-forecast` @ 0.15.8
  - URL: https://www.npmjs.com/package/@sipemu/anofox-forecast
  - Auth: OIDC (OpenID Connect) via GitHub trusted publisher (no secrets in CI)
  - Provenance: Automatically attested via `npm publish --provenance` (npm 11+)
  - Access: Public with scope (`@sipemu`)

**Documentation Hosting:**
- docs.rs - Auto-generated Rust API documentation
  - URL: https://docs.rs/anofox-forecast
  - Build trigger: Automatic on crates.io release
  - Quality gate: Build fails if rustdoc has warnings (`RUSTDOCFLAGS: -D warnings`)

**Code Repository:**
- GitHub - Source control and CI/CD hosting
  - Repository: https://github.com/sipemu/anofox-forecast
  - Hosting: GitHub Pages for interactive playground
  - URL: https://sipemu.github.io/anofox-forecast/

**Related Integrations (Documented in README):**
- DuckDB extension - SQL-native forecasting at scale (separate repo: https://github.com/DataZooDE/anofox-forecast)
- Interactive browser app - https://muon-stat.com/apps/anofox-app/
- fdars-core crate - Periodicity detection (imported dependency)
- changepoint.forecast - Upstream reference (research package)
- dependence-forecastability - Upstream reference (research package)

## Data Storage

**Databases:**
- None - This is a library crate with no persistent storage layer
- **Example data:** Test fixtures stored locally in `tests/data/` (excluded from published tarball)
- **Validation datasets:** Large benchmark datasets in `validation/data/` (CSV/TSF/JSON/Parquet — excluded from crates.io; users fetch independently per example doc comments)

**File Storage:**
- Local filesystem only — No cloud storage integration
- Generated artifacts:
  - WASM binaries: `js/`, `js-node/`, `js-bundler/` (local output from wasm-pack)
  - Playground assets: `crates/anofox-forecast-js/playground/` (includes HTML/JS/WASM)

**Caching:**
- None configured - No external cache service
- Local build cache: `target/` directory and `~/.cargo/registry` via GitHub Actions workflow caching

## Authentication & Identity

**Auth Provider:**
- GitHub - Source control and CI/CD identity
  - Workflow authentication: Standard GitHub token (auto-generated per job)
  - npm publishing: OIDC federation (OpenID Connect trusted publisher)
  - Pages deployment: Auto-authenticated via GitHub Actions `pages` permission

**No Application-Level Auth:**
- Library is stateless — No user authentication
- All credentials for publishing are ephemeral GitHub Actions secrets/tokens (no .env files committed)

## Monitoring & Observability

**Error Tracking:**
- None configured - Library users handle error tracking for their applications

**Code Coverage:**
- codecov.io - Coverage metrics collection
  - Config: `codecov.yml` (minimal — auto-upload via CI badge)
  - Target: Auto (project default)
  - Patch threshold: 70% minimum
  - Integration: Automatic via GitHub Actions after test runs (badge in README)

**Logs:**
- console/stderr only in examples
- Users control logging in application code (library is silent by default)
- Diagnostic output available via examples (e.g., `cross_validation.rs`, `diagnostics.rs`)

**Health Checks:**
- CI/CD gates:
  - cargo test (all features)
  - cargo clippy (deny warnings)
  - cargo fmt (deny format violations)
  - cargo audit (ignore allowlist for transitive advisories)
  - cargo deny (license and source scanning)
  - rustdoc build (deny documentation warnings)
  - WASM compilation check

## CI/CD & Deployment

**Hosting:**
- GitHub Actions - CI/CD and automated publishing
  - Runners: ubuntu-latest (standard GH runner)
  - Triggered by:
    - Push to main/master
    - Pull requests
    - Release published events
    - Workflow dispatch (manual trigger)

**CI Pipeline:**
- `.github/workflows/ci.yml` - Continuous integration
  - Runs on: Push to main/master, all PRs, release published
  - Jobs:
    - Test: `cargo test --all-features` across stable/beta/nightly Rust
    - Clippy: Linting with `-D warnings`
    - Format: `cargo fmt --all -- --check`
    - Audit: `cargo audit` with RUSTSEC allowlist
    - Deny: `cargo deny check` (licenses, sources)
    - Docs: `cargo doc --no-deps --all-features`
    - WASM: `cargo build -p anofox-forecast-js --target wasm32-unknown-unknown`

- `.github/workflows/npm.yml` - NPM package publishing
  - Triggered by: Release published or manual workflow_dispatch
  - Jobs:
    - Build WASM: wasm-pack release build for web target
    - Test JS: Node.js runtime tests (`node --test test.mjs`)
    - Test WASM: Headless browser tests (Chrome)
    - Publish: OIDC-authenticated npm publish with provenance
  - Workflow dispatch: Optional dry-run mode
  - Version detection: Reads from `js/package.json` to derive npm dist-tag (alpha/beta/rc/latest)
  - npm version: Pinned to npm@11 to avoid npm@12 sigstore bundle bug

- `.github/workflows/deploy-playground.yml` - Interactive playground to GitHub Pages
  - Triggered by: Push to main touching `crates/anofox-forecast-js/`, manual trigger
  - Deployment: `https://sipemu.github.io/anofox-forecast/`
  - Concurrency: Single deployment at a time (cancels in-flight runs)
  - Pages config: Requires `Settings → Pages → Source = "GitHub Actions"` (one-time setup)

**Artifact Storage:**
- GitHub Actions Artifacts
  - WASM package (1 day retention)
  - Used for multi-job dependency (build → test → publish)

**Caching Strategy:**
- Rust build: `Swatinem/rust-cache@v2` (registry + git + target/)
- Cargo registry: `~/.cargo/registry`, `~/.cargo/git`
- WASM build: Custom cache with Cargo.lock as key

## Environment Configuration

**Required env vars:**
- `CARGO_TERM_COLOR: always` - CI output formatting

**Secrets location:**
- No `.env` files in repository (not applicable for library)
- NPM authentication: GitHub OIDC trusted publisher (no hardcoded tokens)
- GitHub Actions: Uses auto-generated `GITHUB_TOKEN`
- codecov: Auto-integration via Actions (no explicit token needed in modern setup)

**No Configuration Files Required:**
- Deploy-playground: One-time Pages setup in GitHub repository settings
- Everything else auto-configured via CI manifests

## Webhooks & Callbacks

**Incoming:**
- None - This is a library crate (not a service)

**Outgoing:**
- GitHub Actions → npm: Publish trigger on release event
- GitHub Actions → GitHub Pages: Deploy playground on main push
- GitHub → codecov.io: Coverage report submission (via CI step)
- GitHub → docs.rs: Auto-documentation build on crates.io release
- GitHub Releases: Create release trigger (via GitHub UI)

**Dependency Updates:**
- Dependabot: Not explicitly configured in repo
- Security advisories: Monitored via cargo-audit in CI
- Transitive dependencies: cargo-deny tracks multiple versions (warns)

## Security & Attestation

**Security Scanning:**
- cargo-audit: Detects known CVEs (3 transitive allowlisted in deny.toml)
  - RUSTSEC-2024-0436 (paste/criterion/statrs)
  - RUSTSEC-2025-0141 (bincode 1.3 serde feature)
  - RUSTSEC-2026-0204 (crossbeam-utils via rayon/faer)

**License Compliance:**
- cargo-deny: Allows only permissive licenses
  - Allowed: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unicode, Zlib, BSL-1.0, 0BSD, CC0-1.0, OpenSSL, Unlicense
  - Rejects: GPL, AGPL (copyleft)
  - Confidence threshold: 0.8

**Supply Chain:**
- npm provenance: OIDC-attested build artifact signature
- crates.io: Published via HTTPS with Cargo verification
- GitHub: Commit signing optional (not enforced in this repo)

## Cross-Compilation

**WASM Targets:**
- `wasm32-unknown-unknown` - Bare WASM (no runtime)
  - wasm-pack targets:
    - `web` - ES modules for browser (`js/`)
    - `nodejs` - CommonJS for Node.js (`js-node/`)
    - `bundler` - For webpack/rollup (`js-bundler/`)

**Platform Support:**
- x86_64-unknown-linux-gnu (CI/CD)
- aarch64-darwin (not explicitly tested but likely works)
- Windows: Not tested in CI (contributions welcome)

---

*Integration audit: 2026-08-09*
