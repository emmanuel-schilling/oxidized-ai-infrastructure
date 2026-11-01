// reinforcement_learning/tests/discrete_gateway_test.rs

mod common;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;

#[test]
fn test_discrete_initialization_and_observation_bounds() {
    common::ensure_python_initialized();

    let mut env = PyO3DiscreteEnvironment::new("FrozenLake-v1", None)
        .expect("FAIL: Failed to instantiate FrozenLake-v1");

    let initial_state = env
        .reset(Some(42))
        .expect("FAIL: env.reset() threw an error");

    // FrozenLake 4x4 state space: 0 to 15. Initial state is 0.
    assert_eq!(initial_state, 0, "FAIL: FrozenLake start state should be 0");
}

#[test]
fn test_discrete_kwargs_deterministic_transition() {
    common::ensure_python_initialized();

    // Passing `is_slippery: false` creates deterministic transitions
    let mut env = Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_slippery", false).unwrap();
        PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
    })
    .expect("FAIL: Could not create FrozenLake with kwargs");

    let start_state = env.reset(Some(42)).unwrap();
    assert_eq!(start_state, 0);

    // FrozenLake action mappings: 0: Left, 1: Down, 2: Right, 3: Up
    // Moving Right (action = 2) from state 0 deterministically transitions to state 1
    let (next_state, reward, terminated, truncated) =
        env.step(2).expect("FAIL: Step execution failed");

    assert_eq!(
        next_state, 1,
        "FAIL: Non-slippery FrozenLake failed deterministic transition (0 -> 1)"
    );
    assert_eq!(reward, 0.0, "FAIL: Intermediate step should yield 0 reward");
    assert!(!terminated, "FAIL: Step 1 terminated unexpectedly");
    assert!(!truncated, "FAIL: Step 1 truncated unexpectedly");
}

#[test]
fn test_discrete_invalid_environment_name_returns_error() {
    common::ensure_python_initialized();

    let result = PyO3DiscreteEnvironment::new("NonExistentDiscreteEnv-v999", None);
    assert!(
        result.is_err(),
        "FAIL: Invalid env name should return Result::Err"
    );
}

#[test]
fn test_discrete_from_raw_constructor() {
    common::ensure_python_initialized();

    let raw_env = Python::with_gil(|py| {
        let gym = py.import("gymnasium").unwrap();
        gym.call_method1("make", ("FrozenLake-v1",))
            .unwrap()
            .unbind()
    });

    let mut env = PyO3DiscreteEnvironment::from_raw(raw_env);
    let obs = env.reset(None);
    assert!(
        obs.is_ok(),
        "FAIL: from_raw discrete environment failed to reset"
    );
    assert!(obs.unwrap() < 16);
}

#[test]
fn test_discrete_full_contract_lifecycle() {
    common::ensure_python_initialized();

    let mut env = Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_slippery", false).unwrap();
        PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
    })
    .unwrap();

    // Action 1 is Down (from 0 -> 4 -> 8 -> 12)
    let summary = common::validate_environment_contract(&mut env, |_obs| 1, 20, Some(42));

    assert!(summary.total_steps > 0, "FAIL: No steps were executed");
}
