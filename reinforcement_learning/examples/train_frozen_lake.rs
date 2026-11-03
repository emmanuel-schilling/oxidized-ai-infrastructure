// reinforcement_learning/examples/train_frozen_lake.rs

//! Trains the Rust Q-learning agent on slippery FrozenLake, on the native Rust
//! lake or on Gymnasium's through PyO3:
//!
//! ```text
//! cargo run --release --example train_frozen_lake -- \
//!     [--lake native|gymnasium] [--episodes N] [--seed S] [--evaluate]
//! ```
//!
//! Defaults: the native lake, 20,000 episodes, seed 0. The settings match the
//! equivalence tests: seed `S` seeds the agent with `S` and the environment
//! with 1000 + `S` on the first reset. Prints the episode and step counts as
//! JSON, plus, with `--evaluate`, the trained greedy policy's exact probability
//! of reaching the goal within the step limit.
//!
//! For `gymnasium`, run it through `uv run` (or with a virtual environment
//! active) so the embedded Python finds the environment's packages.

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
const USAGE: &str =
    "usage: train_frozen_lake [--lake native|gymnasium] [--episodes N] [--seed S] [--evaluate]";

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
    evaluate: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        lake: Lake::Native,
        episodes: 20_000,
        seed: 0,
        evaluate: false,
    };
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        if flag == "--evaluate" {
            parsed.evaluate = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--lake" => {
                parsed.lake = match value.as_str() {
                    "native" => Lake::Native,
                    "gymnasium" => Lake::Gymnasium,
                    other => {
                        return Err(format!(
                            "unknown lake {other:?}, expected native or gymnasium"
                        ));
                    }
                }
            }
            "--episodes" => {
                parsed.episodes = value
                    .parse()
                    .map_err(|e| format!("invalid episode count {value:?}: {e}"))?;
            }
            "--seed" => {
                parsed.seed = value
                    .parse()
                    .map_err(|e| format!("invalid seed {value:?}: {e}"))?;
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    Ok(parsed)
}

/// Trains a fresh agent for `episodes` and returns it with the total number of steps.
fn train<E>(env: &mut E, episodes: usize, seed: u64) -> Result<(QLearning, usize), Box<dyn Error>>
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
    Ok((agent, steps))
}

/// Embedded Python sees only its base installation's packages. When a virtual
/// environment is active, as under `uv run`, adds its site-packages to
/// `sys.path`. Python's `sysconfig` locates it with its `venv` scheme (Python
/// 3.11+), which gives the right layout on every platform; the default scheme
/// would not, for example on a macOS framework build.
fn add_virtual_environment(py: Python<'_>) -> PyResult<()> {
    let Some(venv) = std::env::var_os("VIRTUAL_ENV") else {
        return Ok(());
    };
    let vars = PyDict::new(py);
    vars.set_item("base", &venv)?;
    vars.set_item("platbase", &venv)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("vars", vars)?;
    let site_packages = py
        .import("sysconfig")?
        .call_method("get_paths", ("venv",), Some(&kwargs))?
        .get_item("purelib")?;
    py.import("sys")?
        .getattr("path")?
        .call_method1("insert", (0, site_packages))?;
    Ok(())
}

/// Trains as `args` asks and returns the step count and, if asked, the
/// greedy policy's success probability.
fn run(args: &Args) -> Result<(usize, Option<f64>), Box<dyn Error>> {
    // Both lakes share Gymnasium's transition table (see the parity tests), so
    // the native one scores agents trained on either.
    let lake = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))?;
    let (agent, steps) = match args.lake {
        Lake::Native => {
            let mut env = FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS))?;
            train(&mut env, args.episodes, args.seed)?
        }
        Lake::Gymnasium => {
            let mut env = Python::with_gil(|py| {
                add_virtual_environment(py)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("is_slippery", true)?;
                PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
            })?;
            train(&mut env, args.episodes, args.seed)?
        }
    };
    let success = if args.evaluate {
        Some(lake.success_probability(&agent.greedy_policy(), DEFAULT_MAX_EPISODE_STEPS)?)
    } else {
        None
    };
    Ok((steps, success))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = parse_args(&args)
        .map_err(Box::<dyn Error>::from)
        .and_then(|args| run(&args).map(|outcome| (args, outcome)));
    match result {
        Ok((args, (steps, success))) => {
            let evaluation = success
                .map(|p| format!(", \"success_probability\": {p}"))
                .unwrap_or_default();
            println!(
                "{{\"episodes\": {}, \"steps\": {steps}{evaluation}}}",
                args.episodes
            );
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
    fn parses_named_arguments_with_defaults() {
        assert_eq!(
            parse_args(&[]),
            Ok(Args {
                lake: Lake::Native,
                episodes: 20_000,
                seed: 0,
                evaluate: false
            })
        );
        assert_eq!(
            parse_args(&strings(&[
                "--evaluate",
                "--seed",
                "3",
                "--lake",
                "gymnasium",
                "--episodes",
                "500"
            ])),
            Ok(Args {
                lake: Lake::Gymnasium,
                episodes: 500,
                seed: 3,
                evaluate: true
            })
        );
    }

    #[test]
    fn rejects_invalid_arguments() {
        for args in [
            vec!["native"],
            vec!["--lake"],
            vec!["--lake", "cartpole"],
            vec!["--episodes", "-1"],
            vec!["--seed", "seed"],
            vec!["--episodes", "10", "--verbose"],
        ] {
            assert!(parse_args(&strings(&args)).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn native_training_counts_every_step_and_can_be_evaluated() {
        let args = Args {
            lake: Lake::Native,
            episodes: 50,
            seed: 0,
            evaluate: true,
        };
        let (steps, success) = run(&args).expect("training failed");
        // Each episode takes between 1 and 100 steps.
        assert!((50..=5_000).contains(&steps), "{steps} steps");
        let success = success.expect("no evaluation");
        assert!((0.0..=1.0).contains(&success), "{success}");

        // Zero episodes is the startup-only baseline the benchmarks subtract.
        let args = Args {
            episodes: 0,
            evaluate: false,
            ..args
        };
        assert_eq!(run(&args).expect("training failed"), (0, None));
    }
}
