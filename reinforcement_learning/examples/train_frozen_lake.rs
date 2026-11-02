// reinforcement_learning/examples/train_frozen_lake.rs

//! Trains the Rust Q-learning agent on slippery FrozenLake for end-to-end
//! benchmarks, on the native Rust lake or on Gymnasium's through PyO3:
//!
//! ```text
//! cargo run --release --example train_frozen_lake -- <native|gymnasium> <episodes> <seed>
//! ```
//!
//! The settings match the equivalence tests: run `seed` seeds the agent with
//! `seed` and the environment with 1000 + `seed` on the first reset. Prints the
//! episode and step counts as JSON. For `gymnasium`, Python must find the
//! virtual environment's packages, for example through `PYTHONPATH`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::agents::q_learning::{QLearning, QLearningParams};
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::envs::frozen_lake::{DEFAULT_MAX_EPISODE_STEPS, FrozenLake, MAP_4X4};
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;
use std::error::Error;
use std::process::ExitCode;

const PARAMS: QLearningParams = QLearningParams {
    epsilon: 0.1,
    step_size: 0.02,
    discount: 0.9,
};
const ENV_SEED_OFFSET: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Lake {
    Native,
    Gymnasium,
}

#[derive(Debug, PartialEq)]
struct Args {
    lake: Lake,
    episodes: usize,
    seed: u64,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let [lake, episodes, seed] = args else {
        return Err("expected 3 arguments: <native|gymnasium> <episodes> <seed>".into());
    };
    let lake = match lake.as_str() {
        "native" => Lake::Native,
        "gymnasium" => Lake::Gymnasium,
        other => {
            return Err(format!(
                "unknown lake {other:?}, expected native or gymnasium"
            ));
        }
    };
    let episodes = episodes
        .parse()
        .map_err(|e| format!("invalid episode count {episodes:?}: {e}"))?;
    let seed = seed
        .parse()
        .map_err(|e| format!("invalid seed {seed:?}: {e}"))?;
    Ok(Args {
        lake,
        episodes,
        seed,
    })
}

/// Trains a fresh agent for `episodes` and returns the total number of steps.
fn train<E>(env: &mut E, episodes: usize, seed: u64) -> Result<usize, Box<dyn Error>>
where
    E: Environment<Observation = usize, Action = usize>,
    E::Error: Error + 'static,
{
    let mut agent = QLearning::new(16, 4, PARAMS, seed)?;
    let mut steps = 0;
    for episode in 0..episodes {
        let env_seed = (episode == 0).then_some(ENV_SEED_OFFSET + seed);
        steps += agent.train_episode(env, env_seed)?.steps;
    }
    Ok(steps)
}

fn run(args: &Args) -> Result<usize, Box<dyn Error>> {
    match args.lake {
        Lake::Native => {
            let mut env = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))?;
            train(&mut env, args.episodes, args.seed)
        }
        Lake::Gymnasium => {
            let mut env = Python::with_gil(|py| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("is_slippery", true)?;
                PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
            })?;
            train(&mut env, args.episodes, args.seed)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = parse_args(&args)
        .map_err(Box::<dyn Error>::from)
        .and_then(|args| run(&args).map(|steps| (args, steps)));
    match result {
        Ok((args, steps)) => {
            println!("{{\"episodes\": {}, \"steps\": {steps}}}", args.episodes);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("train_frozen_lake: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn parses_valid_arguments() {
        assert_eq!(
            parse_args(&strings(&["gymnasium", "20000", "3"])),
            Ok(Args {
                lake: Lake::Gymnasium,
                episodes: 20_000,
                seed: 3
            })
        );
    }

    #[test]
    fn rejects_invalid_arguments() {
        for args in [
            vec![],
            vec!["native", "10"],
            vec!["native", "10", "0", "extra"],
            vec!["cartpole", "10", "0"],
            vec!["native", "-1", "0"],
            vec!["native", "10", "seed"],
        ] {
            assert!(parse_args(&strings(&args)).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn native_training_counts_every_step() {
        let args = Args {
            lake: Lake::Native,
            episodes: 50,
            seed: 0,
        };
        // Each episode takes between 1 and 100 steps.
        let steps = run(&args).expect("training failed");
        assert!((50..=5_000).contains(&steps), "{steps} steps");
        // Zero episodes is the startup-only baseline the benchmarks subtract.
        let args = Args {
            episodes: 0,
            ..args
        };
        assert_eq!(run(&args).expect("training failed"), 0);
    }
}
