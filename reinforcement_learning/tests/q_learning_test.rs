// reinforcement_learning/tests/q_learning_test.rs

mod common;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::agents::q_learning::{QLearning, QLearningParams};
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::envs::frozen_lake::{DEFAULT_MAX_EPISODE_STEPS, FrozenLake, MAP_4X4};
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;

const PARAMS: QLearningParams = QLearningParams {
    epsilon: 0.1,
    step_size: 0.5,
    discount: 0.9,
};
const AGENT_SEED: u64 = 11;
const ENV_SEED: u64 = 2024;

fn rust_lake(is_slippery: bool) -> FrozenLake {
    FrozenLake::new(&MAP_4X4, is_slippery, Some(DEFAULT_MAX_EPISODE_STEPS))
        .expect("FAIL: Could not build the Rust FrozenLake")
}

/// Trains a fresh agent for `episodes`, seeding the environment on the first reset only.
fn train<E>(env: &mut E, params: QLearningParams, episodes: usize) -> QLearning
where
    E: Environment<Observation = usize, Action = usize>,
    E::Error: std::fmt::Debug,
{
    let mut agent =
        QLearning::new(16, 4, params, AGENT_SEED).expect("FAIL: Could not build the agent");
    for episode in 0..episodes {
        let seed = (episode == 0).then_some(ENV_SEED);
        agent
            .train_episode(env, seed)
            .expect("FAIL: Training episode returned an error");
    }
    agent
}

/// Runs one greedy episode and returns (steps, total reward).
fn greedy_episode<E>(agent: &mut QLearning, env: &mut E) -> (usize, f64)
where
    E: Environment<Observation = usize, Action = usize>,
    E::Error: std::fmt::Debug,
{
    let mut state = env.reset(None).expect("FAIL: reset() returned an error");
    let mut steps = 0;
    loop {
        let action = agent
            .greedy_action(state)
            .expect("FAIL: greedy_action() returned an error");
        let (next_state, reward, terminated, truncated) =
            env.step(action).expect("FAIL: step() returned an error");
        steps += 1;
        if terminated || truncated {
            return (steps, reward);
        }
        state = next_state;
    }
}

#[test]
fn test_q_learning_finds_the_shortest_path_on_the_non_slippery_lake() {
    let mut env = rust_lake(false);
    let mut agent = train(&mut env, PARAMS, 500);

    // Every shortest path from 0 to 15 on the 4x4 map takes 6 steps.
    assert_eq!(greedy_episode(&mut agent, &mut env), (6, 1.0));
}

#[test]
fn test_q_learning_hybrid_and_pure_rust_learn_identical_tables_when_deterministic() {
    common::ensure_python_initialized();

    let mut gym_env = Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_slippery", false)?;
        PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
    })
    .expect("FAIL: Could not create Gymnasium FrozenLake");
    let mut rust_env = rust_lake(false);

    // Without slipping, neither environment uses randomness after reset, so the
    // agent's own seeded stream fully determines both runs.
    let hybrid = train(&mut gym_env, PARAMS, 200);
    let pure = train(&mut rust_env, PARAMS, 200);

    // Guards against a vacuous match between two untrained, all-zero tables.
    let goal_value = pure
        .q_values(14)
        .expect("FAIL: Q-table is missing state 14")[2];
    assert!(goal_value > 0.0, "FAIL: The goal was never reached");
    for state in 0..16 {
        assert_eq!(
            hybrid.q_values(state),
            pure.q_values(state),
            "FAIL: Q-values differ at state {state}"
        );
    }
}

#[test]
fn test_q_learning_beats_a_random_policy_on_the_slippery_lake() {
    const EVAL_EPISODES: usize = 1_000;
    let mut env = rust_lake(true);
    // A constant step size of 0.5 keeps chasing individual slips; 0.02 averages
    // over them, so the greedy policy settles near the optimum.
    let params = QLearningParams {
        step_size: 0.02,
        ..PARAMS
    };
    let mut agent = train(&mut env, params, 20_000);

    let successes = (0..EVAL_EPISODES)
        .filter(|_| greedy_episode(&mut agent, &mut env).1 > 0.0)
        .count();
    let rate = successes as f64 / EVAL_EPISODES as f64;
    // A uniform random policy succeeds about 1.5% of the time; the optimal
    // policy for discount 0.9 about 73% (value iteration on Gymnasium's table).
    assert!(
        rate > 0.3,
        "FAIL: Greedy success rate {rate} after training"
    );
}
