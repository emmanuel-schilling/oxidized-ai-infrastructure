// reinforcement_learning/src/pyo3_gateways/error.rs

use pyo3::PyErr;
use std::fmt;

/// Error type capturing failures across the Python/Gymnasium FFI boundary
#[derive(Debug)]
pub enum GatewayError {
    PyError(PyErr),
    StructureMismatch(String),
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PyError(err) => write!(f, "Python FFI Exception: {err}"),
            Self::StructureMismatch(msg) => write!(f, "FFI Data Layout Mismatch: {msg}"),
        }
    }
}

impl std::error::Error for GatewayError {}

impl From<PyErr> for GatewayError {
    fn from(err: PyErr) -> Self {
        Self::PyError(err)
    }
}
