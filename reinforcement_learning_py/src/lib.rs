// reinforcement_learning_py/src/lib.rs

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use reinforcement_learning::agents::q_learning::{QLearning, QLearningError, QLearningParams};

/// Maps agent errors to the exceptions the pure-Python agent raises.
fn to_py_err(err: QLearningError) -> PyErr {
    match err {
        QLearningError::InvalidParameter(_) => PyValueError::new_err(err.to_string()),
        QLearningError::InvalidState(_) | QLearningError::InvalidAction(_) => {
            PyIndexError::new_err(err.to_string())
        }
    }
}

/// Converts a Python int to a table index. Taking `isize` rather than `usize`
/// makes negative values raise IndexError, as in the pure-Python agent,
/// instead of OverflowError.
fn to_index(value: isize, kind: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyIndexError::new_err(format!("{kind} {value} is outside the Q-table")))
}

/// The Rust tabular Q-learning agent, with the same interface as the
/// pure-Python `q_learning.QLearningAgent`.
#[pyclass(name = "QLearningAgent", module = "oxidized_rl")]
struct PyQLearningAgent {
    inner: QLearning,
}

#[pymethods]
impl PyQLearningAgent {
    /// `params` is any object with `epsilon`, `step_size` and `discount`
    /// attributes, such as `q_learning.QLearningParams`.
    #[new]
    fn new(
        num_states: usize,
        num_actions: usize,
        params: &Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Self> {
        let params = QLearningParams {
            epsilon: params.getattr("epsilon")?.extract()?,
            step_size: params.getattr("step_size")?.extract()?,
            discount: params.getattr("discount")?.extract()?,
        };
        QLearning::new(num_states, num_actions, params, seed)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    fn q_values(&self, state: isize) -> PyResult<Vec<f64>> {
        self.inner
            .q_values(to_index(state, "State")?)
            .map(<[f64]>::to_vec)
            .map_err(to_py_err)
    }

    fn greedy_action(&mut self, state: isize) -> PyResult<usize> {
        self.inner
            .greedy_action(to_index(state, "State")?)
            .map_err(to_py_err)
    }

    fn select_action(&mut self, state: isize) -> PyResult<usize> {
        self.inner
            .select_action(to_index(state, "State")?)
            .map_err(to_py_err)
    }

    fn update(
        &mut self,
        state: isize,
        action: isize,
        reward: f64,
        next_state: isize,
        terminated: bool,
    ) -> PyResult<f64> {
        self.inner
            .update(
                to_index(state, "State")?,
                to_index(action, "Action")?,
                reward,
                to_index(next_state, "State")?,
                terminated,
            )
            .map_err(to_py_err)
    }
}

#[pymodule]
fn oxidized_rl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyQLearningAgent>()
}
