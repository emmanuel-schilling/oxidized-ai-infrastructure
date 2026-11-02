// reinforcement_learning/benches/environment.rs

//! Microbenchmarks of one slippery FrozenLake step, on the native Rust lake and
//! on Gymnasium's through PyO3. Episodes that end are reset inside the timed
//! loop, as in training. Python must find the virtual environment's packages,
//! for example through `PYTHONPATH`.

use criterion::{Criterion, criterion_group, criterion_main};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use reinforcement_learning::core::environment::Environment;
use reinforcement_learning::envs::frozen_lake::{DEFAULT_MAX_EPISODE_STEPS, FrozenLake, MAP_4X4};
use reinforcement_learning::pyo3_gateways::discrete::PyO3DiscreteEnvironment;
use std::fmt::Debug;
use std::hint::black_box;

fn bench_steps<E>(c: &mut Criterion, name: &str, env: &mut E)
where
    E: Environment<Observation = usize, Action = usize>,
    E::Error: Debug,
{
    env.reset(Some(0)).expect("reset");
    let mut action = 0;
    c.bench_function(name, |b| {
        b.iter(|| {
            action = (action + 1) % 4;
            let (_, _, terminated, truncated) = env.step(black_box(action)).expect("step");
            if terminated || truncated {
                env.reset(None).expect("reset");
            }
        })
    });
}

fn environment(c: &mut Criterion) {
    let mut native =
        FrozenLake::new(&MAP_4X4, true, Some(DEFAULT_MAX_EPISODE_STEPS)).expect("native lake");
    bench_steps(c, "native_lake/step", &mut native);

    let mut gymnasium = Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_slippery", true)?;
        PyO3DiscreteEnvironment::new("FrozenLake-v1", Some(&kwargs))
    })
    .expect("Gymnasium lake");
    bench_steps(c, "gymnasium_lake_via_pyo3/step", &mut gymnasium);
}

criterion_group!(benches, environment);
criterion_main!(benches);
