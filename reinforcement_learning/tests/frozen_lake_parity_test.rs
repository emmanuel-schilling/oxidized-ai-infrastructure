// reinforcement_learning/tests/frozen_lake_parity_test.rs

mod common;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::envs::frozen_lake::{
    DEFAULT_MAX_EPISODE_STEPS, DOWN, FrozenLake, MAP_4X4,
};
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;
use std::collections::HashMap;

/// Gymnasium's transition table: state -> action -> [(probability, next_state, reward, terminated)]
type GymTable = HashMap<usize, HashMap<usize, Vec<(f64, usize, f64, bool)>>>;

fn gym_map(name: &str) -> Vec<String> {
    Python::with_gil(|py| {
        py.import("gymnasium.envs.toy_text.frozen_lake")
            .and_then(|module| module.getattr("MAPS"))
            .and_then(|maps| maps.get_item(name))
            .and_then(|map| map.extract())
            .expect("FAIL: Could not read Gymnasium's FrozenLake MAPS")
    })
}

/// Builds a Gymnasium FrozenLake and returns its transition table and start-state distribution.
fn gym_lake(map_name: &str, is_slippery: bool) -> (GymTable, Vec<f64>) {
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("map_name", map_name).unwrap();
        kwargs.set_item("is_slippery", is_slippery).unwrap();
        let env = py
            .import("gymnasium")
            .and_then(|gym| gym.call_method("make", ("FrozenLake-v1",), Some(&kwargs)))
            .and_then(|env| env.getattr("unwrapped"))
            .expect("FAIL: Could not create Gymnasium FrozenLake");
        let table: GymTable = env
            .getattr("P")
            .and_then(|p| p.extract())
            .expect("FAIL: Could not extract Gymnasium's transition table");
        let start: Vec<f64> = env
            .getattr("initial_state_distrib")
            .and_then(|d| d.call_method0("tolist"))
            .and_then(|d| d.extract())
            .expect("FAIL: Could not extract the start-state distribution");
        (table, start)
    })
}

#[test]
fn test_frozen_lake_default_map_matches_gymnasium() {
    common::ensure_python_initialized();
    assert_eq!(gym_map("4x4"), MAP_4X4);
}

#[test]
fn test_frozen_lake_step_limit_matches_gymnasium_registration() {
    common::ensure_python_initialized();

    let max_steps: usize = Python::with_gil(|py| {
        py.import("gymnasium")
            .and_then(|gym| gym.call_method1("spec", ("FrozenLake-v1",)))
            .and_then(|spec| spec.getattr("max_episode_steps"))
            .and_then(|steps| steps.extract())
            .expect("FAIL: Could not read FrozenLake-v1's registered step limit")
    });
    assert_eq!(max_steps, DEFAULT_MAX_EPISODE_STEPS);
}

#[test]
fn test_frozen_lake_transition_tables_match_gymnasium_exactly() {
    common::ensure_python_initialized();

    for map_name in ["4x4", "8x8"] {
        let map = gym_map(map_name);
        let rows: Vec<&str> = map.iter().map(String::as_str).collect();
        for is_slippery in [false, true] {
            let case = format!("map {map_name}, slippery {is_slippery}");
            let (gym_table, gym_start) = gym_lake(map_name, is_slippery);
            let mut env = FrozenLake::new(&rows, is_slippery, Some(DEFAULT_MAX_EPISODE_STEPS))
                .expect("FAIL: Could not build the Rust FrozenLake");

            assert_eq!(
                env.num_states(),
                gym_table.len(),
                "FAIL: {case}: state count"
            );
            for (&state, by_action) in &gym_table {
                assert_eq!(
                    by_action.len(),
                    env.num_actions(),
                    "FAIL: {case}: action count"
                );
                for (&action, gym_outcomes) in by_action {
                    let rust_outcomes: Vec<(f64, usize, f64, bool)> = env
                        .transitions(state, action)
                        .expect("FAIL: Rust table is missing an entry")
                        .iter()
                        .map(|t| (t.probability, t.next_state, t.reward, t.terminated))
                        .collect();
                    // Exact float equality holds: both sides compute 1.0 / 3.0 in IEEE doubles.
                    assert_eq!(
                        &rust_outcomes, gym_outcomes,
                        "FAIL: {case}: state {state}, action {action}"
                    );
                }
            }

            // Both built-in maps have a single start tile, so reset is deterministic.
            let gym_starts: Vec<usize> = (0..gym_start.len())
                .filter(|&s| gym_start[s] > 0.0)
                .collect();
            assert_eq!(
                gym_starts.len(),
                1,
                "FAIL: {case}: expected one start state"
            );
            for seed in 0..5 {
                assert_eq!(
                    env.reset(Some(seed)),
                    Ok(gym_starts[0]),
                    "FAIL: {case}: start state"
                );
            }
        }
    }
}

struct EpisodeStats {
    successes: usize,
    lengths: Vec<f64>,
}

/// Runs uniformly random episodes, with the policy's RNG separate from the environment's.
fn run_random_episodes<E>(env: &mut E, episodes: usize, seed: u64) -> EpisodeStats
where
    E: Environment<Observation = usize, Action = usize>,
{
    let mut policy = ChaCha8Rng::seed_from_u64(seed);
    let mut stats = EpisodeStats {
        successes: 0,
        lengths: Vec::with_capacity(episodes),
    };
    env.reset(Some(seed))
        .expect("FAIL: Initial seeded reset() returned an error");
    for episode in 0..episodes {
        if episode > 0 {
            env.reset(None).expect("FAIL: reset() returned an error");
        }
        let mut length = 0;
        loop {
            let (_, reward, terminated, truncated) = env
                .step(policy.gen_range(0..4))
                .expect("FAIL: step() returned an error");
            length += 1;
            if terminated || truncated {
                if reward > 0.0 {
                    stats.successes += 1;
                }
                break;
            }
        }
        stats.lengths.push(length as f64);
    }
    stats
}

fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance)
}

#[test]
fn test_frozen_lake_random_episodes_match_gymnasium_statistically() {
    common::ensure_python_initialized();

    const EPISODES: usize = 20_000;
    const SEED: u64 = 2024;
    let mut gym_env = PyO3DiscreteEnvironment::new("FrozenLake-v1", None)
        .expect("FAIL: Could not create Gymnasium FrozenLake");
    let mut rust_env = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake");

    let gym = run_random_episodes(&mut gym_env, EPISODES, SEED);
    let rust = run_random_episodes(&mut rust_env, EPISODES, SEED);
    let n = EPISODES as f64;

    // Two-proportion z-test on the success rate
    let (p_gym, p_rust) = (gym.successes as f64 / n, rust.successes as f64 / n);
    let pooled = (p_gym + p_rust) / 2.0;
    let z_success = (p_gym - p_rust) / (pooled * (1.0 - pooled) * 2.0 / n).sqrt();
    assert!(
        z_success.abs() < 4.0,
        "FAIL: Success rates differ: Gymnasium {p_gym}, Rust {p_rust}, z = {z_success}"
    );

    // Two-sample z-test on the mean episode length
    let (mean_gym, var_gym) = mean_and_variance(&gym.lengths);
    let (mean_rust, var_rust) = mean_and_variance(&rust.lengths);
    let z_length = (mean_gym - mean_rust) / ((var_gym + var_rust) / n).sqrt();
    assert!(
        z_length.abs() < 4.0,
        "FAIL: Mean episode lengths differ: Gymnasium {mean_gym}, Rust {mean_rust}, z = {z_length}"
    );
}

#[test]
fn test_frozen_lake_full_contract_lifecycle() {
    let mut env = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake");

    let summary = common::validate_environment_contract(&mut env, |_obs| DOWN, 100, Some(42));

    assert!(summary.total_steps > 0, "FAIL: No steps were executed");
    assert!(
        summary.reached_terminal,
        "FAIL: Episode did not end within the step limit"
    );
}
