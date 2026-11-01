// reinforcement_learning/tests/common/mod.rs

use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::core::environment::Environment;
use std::env;
use std::path::PathBuf;
use std::sync::OnceLock;

static INIT_PYTHON: OnceLock<()> = OnceLock::new();

/// Initializes the embedded Python runtime once across all parallel test threads.
/// Dynamically resolves the virtual environment rather than hardcoding a Python minor version.
pub fn ensure_python_initialized() {
    INIT_PYTHON.get_or_init(|| {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let sys = py.import("sys").expect("FAIL: sys module unavailable");
            let path = sys.getattr("path").expect("FAIL: sys.path unavailable");

            // 1. Check if VIRTUAL_ENV environment variable is set
            let mut site_packages_opt = env::var("VIRTUAL_ENV")
                .ok()
                .and_then(|venv| find_site_packages(PathBuf::from(venv)));

            // 2. Fallback: scan current directory and its parent for .venv
            if site_packages_opt.is_none() {
                let current = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let candidates = [
                    current.join(".venv"),
                    current
                        .parent()
                        .map(|p| p.join(".venv"))
                        .unwrap_or_default(),
                ];
                for candidate in candidates {
                    if candidate.exists() {
                        site_packages_opt = find_site_packages(candidate);
                        if site_packages_opt.is_some() {
                            break;
                        }
                    }
                }
            }

            if let Some(site_packages) = site_packages_opt {
                path.call_method1("insert", (0, site_packages.to_string_lossy().to_string()))
                    .expect("FAIL: Could not inject site-packages into sys.path");
            }

            // Import gymnasium fully before any test thread touches it. Concurrent
            // first imports of the package and one of its submodules can trip
            // Python's import-lock deadlock detection, which hands one thread a
            // partially initialized module.
            py.import("gymnasium")
                .expect("FAIL: Could not import gymnasium");
        });
    });
}

/// Dynamically discovers the site-packages directory inside a .venv folder
fn find_site_packages(venv_root: PathBuf) -> Option<PathBuf> {
    let lib_dir = venv_root.join("lib");
    if let Ok(entries) = std::fs::read_dir(lib_dir) {
        for entry in entries.flatten() {
            let site_packages = entry.path().join("site-packages");
            if site_packages.exists() {
                return Some(site_packages);
            }
        }
    }
    None
}

/// Helper to construct a PyDict kwargs handle safely inside a GIL context
#[allow(dead_code)]
pub fn make_kwargs<'py>(
    py: Python<'py>,
    entries: &[(&str, Bound<'py, PyAny>)],
) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    for (k, v) in entries {
        dict.set_item(k, v)
            .expect("FAIL: Failed to insert entry into kwargs dictionary");
    }
    dict
}

/// Asserts that a continuous observation tensor is well-formed
#[allow(dead_code)]
pub fn assert_valid_continuous_observation(obs: &Array1<f64>, expected_len: usize) {
    assert_eq!(
        obs.len(),
        expected_len,
        "FAIL: Observation dimension mismatch: expected {}, got {}",
        expected_len,
        obs.len()
    );

    for (idx, &val) in obs.iter().enumerate() {
        assert!(
            !val.is_nan(),
            "FAIL: Observation component at index {} is NaN",
            idx
        );
        assert!(
            !val.is_infinite(),
            "FAIL: Observation component at index {} is Infinite ({})",
            idx,
            val
        );
    }
}

#[allow(dead_code)]
pub struct ContractCheckSummary {
    pub total_steps: usize,
    pub cumulative_reward: f64,
    pub reached_terminal: bool,
}

/// Validates that any gateway implementation rigorously honors the fallible, seeded Environment contract
#[allow(dead_code)]
pub fn validate_environment_contract<E, F>(
    env: &mut E,
    mut action_sampler: F,
    max_steps: usize,
    seed: Option<u64>,
) -> ContractCheckSummary
where
    E: Environment,
    F: FnMut(&E::Observation) -> E::Action,
{
    let mut current_obs = env
        .reset(seed)
        .expect("FAIL: Initial environment reset() returned an error");
    let mut total_steps = 0;
    let mut cumulative_reward = 0.0;
    let mut reached_terminal = false;

    for _ in 0..max_steps {
        let action = action_sampler(&current_obs);
        let (next_obs, reward, terminated, truncated) = env
            .step(action)
            .expect("FAIL: Environment step() returned an error");

        assert!(
            !reward.is_nan(),
            "FAIL: Environment produced NaN reward scalar"
        );
        assert!(
            !reward.is_infinite(),
            "FAIL: Environment produced infinite reward scalar"
        );

        cumulative_reward += reward;
        total_steps += 1;
        current_obs = next_obs;

        if terminated || truncated {
            reached_terminal = true;
            break;
        }
    }

    // Must be safely re-resettable after termination/truncation
    let post_terminal_obs = env
        .reset(seed)
        .expect("FAIL: Post-terminal environment reset() returned an error");
    let _ = env
        .step(action_sampler(&post_terminal_obs))
        .expect("FAIL: Post-terminal environment step() returned an error");

    ContractCheckSummary {
        total_steps,
        cumulative_reward,
        reached_terminal,
    }
}
