# Model and Data Serialization

**Run:** `cargo run --example serialization --features serde`

## What this example demonstrates

Shows how to serialize and deserialize models, `TimeSeries`, and `Forecast` objects using both JSON (human-readable) and bincode (compact binary) formats. Each round-trip is verified by comparing predictions or values before and after serialization, and a size comparison table highlights the trade-offs between the two formats.

## Sections

1. **Model JSON serialization** -- Serializes a fitted Naive model to JSON, deserializes it, and confirms predictions match the original.
2. **Model bincode serialization** -- Serializes the same model to bincode, compares byte size to JSON, and verifies prediction equality.
3. **TimeSeries serialization** -- Round-trips a `TimeSeries` through both JSON and bincode, confirming values and timestamps are preserved.
4. **Forecast serialization** -- Round-trips a `Forecast` (with lower/upper intervals) through JSON and bincode, verifying equality.
5. **Format size comparison** -- Prints a table comparing JSON and bincode sizes for each object type with compression ratios.

## Key types

- `to_json()` / `from_json()` -- JSON serialization and deserialization
- `to_bincode()` / `from_bincode()` -- bincode serialization and deserialization
- `save_to_file()` / `load_from_file()` -- JSON file I/O (mentioned in summary)
- `save_to_bincode()` / `load_from_bincode()` -- bincode file I/O (mentioned in summary)
- `Naive` / `TimeSeries` / `Forecast` -- types demonstrated in round-trips
