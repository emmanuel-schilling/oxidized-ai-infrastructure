// reinforcement_learning/tests/continuous_gateway_test.rs

mod common;

use pyo3::prelude::*;
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::pyo3_gateways::continuous::PyO3ContinuousEnvironment;

#[test]
fn test_continuous_initialization_and_observation_shape() {
    common::ensure_python_initialized();

    let mut env = PyO3ContinuousEnvironment::new("CartPole-v1", None)
        .expect("FAIL: Failed to instantiate CartPole-v1");

    let obs = env
        .reset(Some(42))
        .expect("FAIL: env.reset() threw an error");

    // CartPole observation space: [cart_pos, cart_velocity, pole_angle, pole_velocity]
    common::assert_valid_continuous_observation(&obs, 4);
}

#[test]
fn test_continuous_seed_determinism() {
    common::ensure_python_initialized();

    let seed = 42;
    let mut env_a = PyO3ContinuousEnvironment::new("CartPole-v1", None).unwrap();
    let mut env_b = PyO3ContinuousEnvironment::new("CartPole-v1", None).unwrap();

    let obs_a = env_a.reset(Some(seed)).unwrap();
    let obs_b = env_b.reset(Some(seed)).unwrap();

    // With identical seeds, initial continuous state vectors must match exactly
    for (idx, (a, b)) in obs_a.iter().zip(obs_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "FAIL: Deterministic seed mismatch at index {idx}: {a} vs {b}"
        );
    }
}

#[test]
fn test_continuous_step_transition() {
    common::ensure_python_initialized();

    let mut env = PyO3ContinuousEnvironment::new("CartPole-v1", None).unwrap();
    let _ = env.reset(Some(123)).unwrap();

    // Push cart right (action = 1)
    let step_result = env.step(1);
    assert!(
        step_result.is_ok(),
        "FAIL: Step execution returned an error"
    );

    let (next_obs, reward, terminated, truncated) = step_result.unwrap();

    common::assert_valid_continuous_observation(&next_obs, 4);
    assert_eq!(
        reward, 1.0,
        "FAIL: CartPole survival step should yield reward 1.0"
    );
    assert!(
        !terminated,
        "FAIL: CartPole terminated on step 1 unexpectedly"
    );
    assert!(
        !truncated,
        "FAIL: CartPole truncated on step 1 unexpectedly"
    );
}

#[test]
fn test_continuous_invalid_environment_name_returns_error() {
    common::ensure_python_initialized();

    // Non-existent environment ID should fail gracefully with GatewayError::PyError
    let result = PyO3ContinuousEnvironment::new("NonExistentEnv-v999", None);
    assert!(
        result.is_err(),
        "FAIL: Invalid env name should return Result::Err"
    );
}

#[test]
fn test_continuous_from_raw_constructor() {
    common::ensure_python_initialized();

    let raw_env = Python::with_gil(|py| {
        let gym = py.import("gymnasium").unwrap();
        gym.call_method1("make", ("CartPole-v1",)).unwrap().unbind()
    });

    let mut env = PyO3ContinuousEnvironment::from_raw(raw_env);
    let obs = env.reset(None);
    assert!(obs.is_ok(), "FAIL: from_raw environment failed to reset");
    assert_eq!(obs.unwrap().len(), 4);
}

#[test]
fn test_continuous_full_contract_lifecycle() {
    common::ensure_python_initialized();

    let mut env = PyO3ContinuousEnvironment::new("CartPole-v1", None).unwrap();

    // Run a policy alternating actions (0, 1) through the contract checker
    let mut toggle = 0;
    let summary = common::validate_environment_contract(
        &mut env,
        |_obs| {
            toggle = 1 - toggle;
            toggle
        },
        50,
        Some(42),
    );

    assert!(summary.total_steps > 0, "FAIL: No steps were executed");
    assert!(
        summary.cumulative_reward > 0.0,
        "FAIL: No reward accumulated"
    );
}
