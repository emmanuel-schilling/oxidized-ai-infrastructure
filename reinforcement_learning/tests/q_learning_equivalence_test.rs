// reinforcement_learning/tests/q_learning_equivalence_test.rs

//! Statistical equivalence of the Rust agent on the native Rust FrozenLake and
//! on Gymnasium's. The two lakes draw slips from different generators, so
//! individual runs differ, but their mean learning curves must agree at every
//! checkpoint.
//!
//! Slow, so opt-in: `cargo test --test q_learning_equivalence_test -- --ignored`.
//! Writes the curves to `results/q_learning_equivalence_rust.json`.

mod common;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::agents::q_learning::{QLearning, QLearningParams};
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::envs::frozen_lake::{DEFAULT_MAX_EPISODE_STEPS, FrozenLake, MAP_4X4};
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;
use std::path::Path;

const PARAMS: QLearningParams = QLearningParams {
    epsilon: 0.1,
    step_size: 0.02,
    discount: 0.9,
};
const EPISODES: usize = 20_000;
const CHECKPOINT_EVERY: usize = 1_000;
const RUNS: u64 = 10;
/// Run `i` seeds the agent with `i` and the environment with `ENV_SEED_OFFSET + i`.
const ENV_SEED_OFFSET: u64 = 1_000;
/// Means may differ by at most this many standard errors of their difference.
/// For correct code, all 20 checkpoints pass with about 99.8% probability
/// (Student's t with about 18 degrees of freedom).
const MAX_STANDARD_ERRORS: f64 = 5.0;

/// The exact probability that the agent's greedy policy reaches the goal from
/// the start state within the step limit. Tied actions share the choice
/// equally, as `greedy_action` does on average, so scoring draws nothing from
/// the agent's generator and leaves training unchanged.
fn success_probability(agent: &QLearning, lake: &FrozenLake) -> f64 {
    // reach[s]: probability of reaching the goal from s within k steps.
    let mut reach = vec![0.0; lake.num_states()];
    for _ in 0..DEFAULT_MAX_EPISODE_STEPS {
        let mut next = vec![0.0; reach.len()];
        for (state, value) in next.iter_mut().enumerate() {
            let q = agent
                .q_values(state)
                .expect("FAIL: Q-table is missing a state");
            let max = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let best: Vec<usize> = (0..q.len()).filter(|&a| q[a] == max).collect();
            for &action in &best {
                for t in lake
                    .transitions(state, action)
                    .expect("FAIL: Lake is missing a transition")
                {
                    let future = if t.terminated {
                        0.0
                    } else {
                        reach[t.next_state]
                    };
                    *value += t.probability * (t.reward + future) / best.len() as f64;
                }
            }
        }
        reach = next;
    }
    reach[0]
}

/// Trains a fresh agent on `env` and scores it every `CHECKPOINT_EVERY` episodes.
fn learning_curve<E>(env: &mut E, lake: &FrozenLake, run: u64) -> Vec<f64>
where
    E: Environment<Observation = usize, Action = usize>,
    E::Error: std::fmt::Debug,
{
    let mut agent = QLearning::new(16, 4, PARAMS, run).expect("FAIL: Could not build the agent");
    let mut curve = Vec::with_capacity(EPISODES / CHECKPOINT_EVERY);
    for episode in 0..EPISODES {
        let seed = (episode == 0).then_some(ENV_SEED_OFFSET + run);
        agent
            .train_episode(env, seed)
            .expect("FAIL: Training episode returned an error");
        if (episode + 1) % CHECKPOINT_EVERY == 0 {
            curve.push(success_probability(&agent, lake));
        }
    }
    curve
}

fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Requires the mean curves to agree within `MAX_STANDARD_ERRORS` at every
/// checkpoint. Comparing whole curves, not a summary such as their mean,
/// catches agents that learn at a different pace but end up similar.
fn assert_curves_agree(a: &[Vec<f64>], b: &[Vec<f64>]) {
    for checkpoint in 0..EPISODES / CHECKPOINT_EVERY {
        let at = |curves: &[Vec<f64>]| -> Vec<f64> {
            curves.iter().map(|curve| curve[checkpoint]).collect()
        };
        let (mean_a, se_a) = mean_and_standard_error(&at(a));
        let (mean_b, se_b) = mean_and_standard_error(&at(b));
        let tolerance = MAX_STANDARD_ERRORS * (se_a.powi(2) + se_b.powi(2)).sqrt();
        assert!(
            (mean_a - mean_b).abs() <= tolerance,
            "FAIL: After {} episodes: native {mean_a:.4} vs Gymnasium {mean_b:.4}, \
             tolerance {tolerance:.4}",
            (checkpoint + 1) * CHECKPOINT_EVERY
        );
    }
}

#[test]
fn test_success_probability_is_one_for_the_shortest_path() {
    let lake = FrozenLake::new(&MAP_4X4, false, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake");
    let mut agent = QLearning::new(16, 4, PARAMS, 0).expect("FAIL: Could not build the agent");
    // Down, down, right, down, right, right: one of the 6-step shortest paths.
    // A terminal update to 1.0 with step size 0.02 gives each step's action
    // the only nonzero value in its row.
    for (state, action) in [(0, 1), (4, 1), (8, 2), (9, 1), (13, 2), (14, 2)] {
        agent
            .update(state, action, 1.0, 0, true)
            .expect("FAIL: update() returned an error");
    }
    assert_eq!(success_probability(&agent, &lake), 1.0);
}

#[test]
fn test_success_probability_of_an_untrained_agent_is_the_random_policys() {
    // An all-zero table ties every action, so the policy is uniformly random,
    // which reaches the goal of the slippery lake about 1.5% of the time.
    let lake = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake");
    let agent = QLearning::new(16, 4, PARAMS, 0).expect("FAIL: Could not build the agent");
    let probability = success_probability(&agent, &lake);
    assert!(
        0.005 < probability && probability < 0.05,
        "FAIL: Random-policy success probability {probability}"
    );
}

#[test]
#[ignore = "slow: trains 20 agents for 20,000 episodes each"]
fn test_native_and_gymnasium_lakes_give_equivalent_learning_curves() {
    common::ensure_python_initialized();
    let lake = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake");

    let mut native = Vec::new();
    let mut gymnasium = Vec::new();
    for run in 0..RUNS {
        let mut native_env = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))
            .expect("FAIL: Could not build the Rust FrozenLake");
        native.push(learning_curve(&mut native_env, &lake, run));

        // The parity tests show both lakes share one transition table, so the
        // native one scores the Gymnasium-trained agents too.
        let mut gym_env = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("is_slippery", true)?;
            PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
        })
        .expect("FAIL: Could not create Gymnasium FrozenLake");
        gymnasium.push(learning_curve(&mut gym_env, &lake, run));
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("FAIL: Crate has no parent directory")
        .join("results");
    std::fs::create_dir_all(&path).expect("FAIL: Could not create the results directory");
    let results = serde_json::json!({
        "description": "Exact greedy success probability within 100 steps on slippery \
                        FrozenLake 4x4, scored every checkpoint_every episodes. Run i \
                        seeds the agent with i and the environment with 1000 + i on the \
                        first reset.",
        "params": {
            "epsilon": PARAMS.epsilon,
            "step_size": PARAMS.step_size,
            "discount": PARAMS.discount,
        },
        "episodes": EPISODES,
        "checkpoint_every": CHECKPOINT_EVERY,
        "curves": {
            "rust_native": native,
            "rust_gymnasium": gymnasium,
        },
    });
    let json = serde_json::to_string_pretty(&results).expect("FAIL: Could not serialize results");
    std::fs::write(path.join("q_learning_equivalence_rust.json"), json + "\n")
        .expect("FAIL: Could not write results");

    assert_curves_agree(&native, &gymnasium);
}
