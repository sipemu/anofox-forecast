# npm release process

The `anofox-forecast-js` WASM package is published to npm automatically when
a GitHub release is published. The workflow lives at
[`.github/workflows/npm.yml`](../.github/workflows/npm.yml).

## Authentication

Publishing uses **npm's OIDC trusted-publisher flow** — no static token,
no `NPM_TOKEN` secret. Auth is bound to the GitHub Actions runner via the
`id-token: write` permission and signed by GitHub's OIDC issuer.

The trust is configured on the npm side (one-time, by the package owner):

- Package: `anofox-forecast-js`
- Repository: `sipemu/anofox-forecast`
- Workflow filename: `npm.yml`
- Permissions: `npm publish`

Configured via **npmjs.com → Package settings → Trusted Publisher → GitHub
Actions**.

## Release flow

1. Bump `crates/anofox-forecast-js/Cargo.toml` **and**
   `crates/anofox-forecast-js/package.json` to the new version. We keep
   them in lock-step with the Rust crate's version.
2. Land the version bump on `main`.
3. Cut a GitHub release on the matching tag (e.g. `v0.7.6`). The release
   event triggers `npm.yml`, which:
   - builds the WASM package via `wasm-pack build --target web`,
   - restores the hand-written `package.json` + `README.md` + `types.d.ts`,
   - runs `npm publish --access public --provenance`.
4. The published artifact appears at
   https://www.npmjs.com/package/anofox-forecast-js with provenance attestation.

## Manual dispatch (dry run)

To verify the pipeline without an actual publish:

```bash
gh workflow run npm.yml --ref main -f dry_run=true
gh run watch
```

The "Publish to npm (dry run)" step should report a successful tarball
prep and no `ENEEDAUTH` / `403` errors.

## Troubleshooting

### `ENEEDAUTH` on publish

- Trusted publisher not configured (or misconfigured) on the npm side
  for this package + workflow. Verify at
  https://www.npmjs.com/package/anofox-forecast-js/access.
- `id-token: write` permission missing from the `publish-npm` job.

### `403 Forbidden`

- Trusted publisher trust record doesn't match the runtime — usually
  caused by a typo in the workflow filename or repo name on the npm
  side. Workflow filename must be exactly `npm.yml`.

### `provenance` failures

- Requires npm ≥ 11.5.1; the workflow's "Upgrade npm for OIDC support"
  step handles this.
- Requires `id-token: write` permission (already set in the workflow).

### Version-mismatch errors

The `package.json` and Rust `Cargo.toml` versions must match. The build
job restores both from git after `wasm-pack` clobbers the
`package.json`, so a mismatch usually means one was bumped and not the
other.
