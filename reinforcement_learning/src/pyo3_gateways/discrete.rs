// reinforcement_learning/src/pyo3_gateways/discrete.rs

use super::GatewayError;
use crate::core::environment::Environment;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// A type-safe wrapper around a dynamic Python Gymnasium environment
/// configured for discrete observations and discrete actions.
pub struct PyO3DiscreteEnvironment {
    gym_env: PyObject,
}

impl PyO3DiscreteEnvironment {
    /// Creates a new discrete environment instance.
    pub fn new(env_name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> Result<Self, GatewayError> {
        Python::with_gil(|py| {
            let gym = py.import("gymnasium")?;
            let env = gym.call_method("make", (env_name,), kwargs)?;
            Ok(Self {
                gym_env: env.into(),
            })
        })
    }

    /// Internal fallback constructor to build the wrapper directly from a raw Python reference pointer.
    pub fn from_raw(obj: PyObject) -> Self {
        Self { gym_env: obj }
    }
}

impl Environment for PyO3DiscreteEnvironment {
    type Observation = usize;
    type Action = usize;
    type Error = GatewayError;

    fn reset(&mut self, seed: Option<u64>) -> Result<Self::Observation, Self::Error> {
        Python::with_gil(|py| {
            let env = self.gym_env.bind(py);

            let result = if let Some(s) = seed {
                let kwargs = PyDict::new(py);
                kwargs.set_item("seed", s)?;
                env.call_method("reset", (), Some(&kwargs))?
            } else {
                env.call_method0("reset")?
            };

            let tuple = result.downcast::<PyTuple>().map_err(|e| {
                GatewayError::StructureMismatch(format!("reset() did not return tuple: {e}"))
            })?;

            let obs: usize = tuple.get_item(0)?.extract().map_err(|e| {
                GatewayError::StructureMismatch(format!("Failed to extract usize obs: {e}"))
            })?;

            Ok(obs)
        })
    }

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<(Self::Observation, f64, bool, bool), Self::Error> {
        Python::with_gil(|py| {
            let env = self.gym_env.bind(py);
            let result = env.call_method1("step", (action,))?;

            let tuple = result.downcast::<PyTuple>().map_err(|e| {
                GatewayError::StructureMismatch(format!("step() did not return tuple: {e}"))
            })?;

            let obs: usize = tuple.get_item(0)?.extract()?;
            let reward: f64 = tuple.get_item(1)?.extract()?;
            let terminated: bool = tuple.get_item(2)?.extract()?;
            let truncated: bool = tuple.get_item(3)?.extract()?;

            Ok((obs, reward, terminated, truncated))
        })
    }
}
