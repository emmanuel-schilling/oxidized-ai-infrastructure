// reinforcement_learning/benches/agent.rs

//! Microbenchmarks of the Rust Q-learning agent's per-step work.

use criterion::{Criterion, criterion_group, criterion_main};
use reinforcement_learning::agents::q_learning::{QLearning, QLearningParams};
use std::hint::black_box;

const PARAMS: QLearningParams = QLearningParams {
    epsilon: 0.1,
    step_size: 0.02,
    discount: 0.9,
};

fn agent(c: &mut Criterion) {
    let mut agent = QLearning::new(16, 4, PARAMS, 0).expect("agent");
    let mut state = 0;
    c.bench_function("rust_agent/select_action", |b| {
        b.iter(|| {
            state = (state + 1) % 16;
            black_box(
                agent
                    .select_action(black_box(state))
                    .expect("select_action"),
            )
        })
    });

    let mut agent = QLearning::new(16, 4, PARAMS, 0).expect("agent");
    let mut step = 0;
    c.bench_function("rust_agent/update", |b| {
        b.iter(|| {
            step = (step + 1) % 64;
            let (state, action) = (step / 4, step % 4);
            black_box(
                agent
                    .update(black_box(state), action, 0.0, (state + 1) % 16, false)
                    .expect("update"),
            )
        })
    });
}

criterion_group!(benches, agent);
criterion_main!(benches);
