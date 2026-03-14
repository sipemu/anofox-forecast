//! Abstract storage layer for orchestration types.
//!
//! Provides a [`PipelineStore`] trait decoupled from any specific serialization
//! format. Future backends (DuckDB, SQLite, etc.) only need to implement this trait.

use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, Utc};

use crate::error::Result;

/// Kind of stored record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    Profile,
    Config,
    Result,
    DecisionLog,
    HorizonAnalysis,
    Report,
}

impl fmt::Display for RecordKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RecordKind::Profile => "profile",
            RecordKind::Config => "config",
            RecordKind::Result => "result",
            RecordKind::DecisionLog => "decision_log",
            RecordKind::HorizonAnalysis => "horizon_analysis",
            RecordKind::Report => "report",
        };
        write!(f, "{}", s)
    }
}

/// A dynamically-typed value for the intermediate representation.
///
/// Intentionally decoupled from serde/JSON — any backend can consume this.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    /// Convenience: create a `Map` from key-value pairs.
    pub fn map_from(pairs: Vec<(&str, Value)>) -> Self {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), v);
        }
        Value::Map(m)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Get a field from a Map value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_map().and_then(|m| m.get(key))
    }
}

/// A single stored record with metadata.
#[derive(Debug, Clone)]
pub struct PipelineRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub kind: RecordKind,
    pub fields: Value,
}

/// Trait for converting orchestration types to/from the storage IR.
pub trait Storable: Sized {
    /// The record kind for this type.
    fn record_kind() -> RecordKind;

    /// Serialize to the IR value.
    fn to_value(&self) -> Value;

    /// Deserialize from the IR value.
    fn from_value(value: &Value) -> Result<Self>;

    /// Create a full record with ID and timestamp.
    fn to_record(&self, id: impl Into<String>) -> PipelineRecord {
        PipelineRecord {
            id: id.into(),
            timestamp: Utc::now(),
            kind: Self::record_kind(),
            fields: self.to_value(),
        }
    }
}

/// Abstract storage backend.
///
/// Implementations might store to JSON files, DuckDB, SQLite, etc.
pub trait PipelineStore {
    /// Save a record. Overwrites if `id` already exists.
    fn save(&self, record: &PipelineRecord) -> Result<()>;

    /// Load a record by ID.
    fn load(&self, id: &str) -> Result<Option<PipelineRecord>>;

    /// List all record IDs, optionally filtered by kind.
    fn list(&self, kind: Option<RecordKind>) -> Result<Vec<String>>;

    /// Delete a record. Returns true if it existed.
    fn delete(&self, id: &str) -> Result<bool>;
}

/// In-memory store for testing.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    records: std::sync::Mutex<BTreeMap<String, PipelineRecord>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PipelineStore for InMemoryStore {
    fn save(&self, record: &PipelineRecord) -> Result<()> {
        let mut store = self.records.lock().unwrap();
        store.insert(record.id.clone(), record.clone());
        Ok(())
    }

    fn load(&self, id: &str) -> Result<Option<PipelineRecord>> {
        let store = self.records.lock().unwrap();
        Ok(store.get(id).cloned())
    }

    fn list(&self, kind: Option<RecordKind>) -> Result<Vec<String>> {
        let store = self.records.lock().unwrap();
        let ids = store
            .iter()
            .filter(|(_, r)| kind.is_none_or(|k| r.kind == k))
            .map(|(id, _)| id.clone())
            .collect();
        Ok(ids)
    }

    fn delete(&self, id: &str) -> Result<bool> {
        let mut store = self.records.lock().unwrap();
        Ok(store.remove(id).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_map_from() {
        let v = Value::map_from(vec![
            ("name", Value::String("test".into())),
            ("count", Value::Int(42)),
        ]);
        assert_eq!(v.get("name").and_then(|v| v.as_str()), Some("test"));
        assert_eq!(v.get("count").and_then(|v| v.as_i64()), Some(42));
    }

    #[test]
    fn value_accessors() {
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Float(2.72).as_f64(), Some(2.72));
        assert_eq!(Value::Int(42).as_i64(), Some(42));
        assert_eq!(Value::Int(42).as_f64(), Some(42.0));
        assert_eq!(Value::String("hi".into()).as_str(), Some("hi"));
        assert!(Value::List(vec![Value::Int(1)]).as_list().is_some());
        assert!(Value::Null.as_str().is_none());
    }

    #[test]
    fn in_memory_store_crud() {
        let store = InMemoryStore::new();
        let record = PipelineRecord {
            id: "test-1".into(),
            timestamp: Utc::now(),
            kind: RecordKind::Config,
            fields: Value::map_from(vec![("horizon", Value::Int(7))]),
        };

        store.save(&record).unwrap();

        let loaded = store.load("test-1").unwrap().unwrap();
        assert_eq!(loaded.id, "test-1");
        assert_eq!(loaded.kind, RecordKind::Config);

        let ids = store.list(None).unwrap();
        assert_eq!(ids, vec!["test-1"]);

        let ids_filtered = store.list(Some(RecordKind::Profile)).unwrap();
        assert!(ids_filtered.is_empty());

        assert!(store.delete("test-1").unwrap());
        assert!(store.load("test-1").unwrap().is_none());
        assert!(!store.delete("test-1").unwrap());
    }

    #[test]
    fn record_kind_display() {
        assert_eq!(format!("{}", RecordKind::Profile), "profile");
        assert_eq!(format!("{}", RecordKind::Config), "config");
        assert_eq!(format!("{}", RecordKind::Result), "result");
    }
}
