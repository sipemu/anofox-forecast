# Makefile for anofox-forecast

.PHONY: all build build-wasm test test-wasm clean install-wasm-pack

# Default target
all: build

# Build the Rust library
build:
	cargo build --release

# Build WASM package
build-wasm: install-wasm-pack
	wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js
	@echo "WASM build complete. Output in js/"

# Build WASM for Node.js
build-wasm-node: install-wasm-pack
	wasm-pack build crates/anofox-forecast-js --target nodejs --out-dir ../../js-node --out-name anofox_forecast_js
	@echo "WASM Node.js build complete. Output in js-node/"

# Build WASM for bundlers (webpack, etc.)
build-wasm-bundler: install-wasm-pack
	wasm-pack build crates/anofox-forecast-js --target bundler --out-dir ../../js-bundler --out-name anofox_forecast_js
	@echo "WASM bundler build complete. Output in js-bundler/"

# Run all tests
test:
	cargo test

# Run WASM tests (requires Chrome)
test-wasm: install-wasm-pack
	wasm-pack test --headless --chrome crates/anofox-forecast-js

# Clean build artifacts
clean:
	cargo clean
	rm -rf js/*.wasm js/*.js js/*.d.ts js/package.json.bak
	rm -rf js-node js-bundler

# Install wasm-pack if not present
install-wasm-pack:
	@which wasm-pack > /dev/null || (echo "Installing wasm-pack..." && curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh)

# Check WASM compilation (for CI)
check-wasm:
	cargo build -p anofox-forecast-js --target wasm32-unknown-unknown

# Format code
fmt:
	cargo fmt --all

# Lint code
clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Publish to npm (requires npm login)
publish-npm: build-wasm
	cd js && npm publish --access public

# Development build (faster, debug)
dev-wasm: install-wasm-pack
	wasm-pack build crates/anofox-forecast-js --target web --out-dir ../../js --out-name anofox_forecast_js --dev
