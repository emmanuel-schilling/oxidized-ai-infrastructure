# oxidized-ai-infrastructure

[![CI](https://github.com/me-like-coffee/oxidized-ai-infrastructure/actions/workflows/ci.yml/badge.svg)](https://github.com/me-like-coffee/oxidized-ai-infrastructure/actions/workflows/ci.yml)

A Rust foundation for implementing reinforcement learning algorithms from the
University of Alberta
[Reinforcement Learning specialization](https://www.coursera.org/specializations/reinforcement-learning)
on Coursera. Environments come from Python's
[Gymnasium](https://gymnasium.farama.org/) and are stepped from Rust through
[PyO3](https://pyo3.rs/).

**Scope:** this is a personal learning project, not a competitive offering.
Public Rust crates already cover similar ground, among them
[`rurel`](https://crates.io/crates/rurel) (tabular Q-learning),
[`border`](https://crates.io/crates/border) (an RL library) and
[`gym-rs`](https://crates.io/crates/gym-rs) (Gym environments in pure Rust).
The aim here is to understand the algorithms and the Rust–Python boundary by
building them.

**Status:** early stage. The environment layer is complete and tested; no RL
algorithms are implemented yet.

## What's implemented

- **`Environment` trait** ([`core/environment.rs`](reinforcement_learning/src/core/environment.rs)):
  a seeded, fallible `reset`/`step` contract with associated observation,
  action and error types, so algorithms never depend on where an environment
  runs.
- **PyO3 gateways** ([`pyo3_gateways/`](reinforcement_learning/src/pyo3_gateways/)):
  wrappers that implement the trait over a `gymnasium.make(...)` environment.
  - `PyO3DiscreteEnvironment`: `usize` observations and actions (e.g. FrozenLake).
  - `PyO3ContinuousEnvironment`: `ndarray::Array1<f64>` observations and
    `usize` actions (e.g. CartPole).

  Python exceptions and unexpected return shapes surface as a typed
  `GatewayError` rather than panics.
- **Integration tests** ([`tests/`](reinforcement_learning/tests/)) against an
  embedded Python interpreter, covering:
  - seeded determinism and known transitions for both gateways
  - a reusable contract checker for any `Environment` implementation
  - the FFI boundary itself: NumPy `float32` to `f64` conversion, GIL
    contention across threads, and exception propagation

## Architecture

```
  Rust                                        Python
 ┌───────────────────────────┐  PyO3      ┌──────────────────────┐
 │ algorithm (planned)       │            │                      │
 │   │ uses                  │ ─ action ▶ │ gymnasium.Env.step() │
 │   ▼                       │            │                      │
 │ Environment trait         │ ◀ obs, r,  │                      │
 │   PyO3*Environment        │   flags ── │                      │
 └───────────────────────────┘            └──────────────────────┘
```

Each `reset` and `step` call acquires the GIL, calls into Gymnasium, and copies
the result into Rust-owned values. The Gymnasium `info` dict is not passed
through.

### Repository layout

```
.
├── reinforcement_learning/
│   ├── src/core/            # Environment trait
│   ├── src/pyo3_gateways/   # Gymnasium wrappers and GatewayError
│   ├── tests/               # integration and FFI boundary tests
│   ├── examples/            # training program for end-to-end benchmarks
│   ├── benches/             # criterion microbenchmarks
│   └── python/              # pure-Python agents, benchmark harness, pytest tests
├── reinforcement_learning_py/ # Rust agents as the Python module oxidized_rl
├── pyproject.toml           # Python dependencies, tool config, maturin build (uv)
├── Dockerfile               # Rust + Python test image
├── docker-compose.yml       # runs the test suite in the container
├── run.sh                   # task runner: dev-test, build, test
└── .github/workflows/ci.yml # CI: build and test on Ubuntu
```

## Getting started

### Prerequisites

Either Docker, or:
- Rust 1.86 or later (the crate uses edition 2024, and its benchmark library,
  criterion, needs 1.86)
- Python 3.11 and [uv](https://docs.astral.sh/uv/); `uv sync` creates a
  `.venv` at the repository root from the pinned versions in `uv.lock` and
  builds the Rust extension module into it

### Build and test

```bash
./run.sh dev-test   # run the tests on the host against .venv
./run.sh build      # build the Docker test image
./run.sh test       # run the tests inside the container
```

The slow equivalence tests are opt-in. They train the Q-learning agent ten
times per variant and check that the mean learning curves agree at every
checkpoint, then write the curves to `results/` as JSON:

```bash
cargo test --test q_learning_equivalence_test -- --ignored
uv run pytest -m slow
```

CI runs the Python checks, the Rust tests and the slow equivalence tests on
Ubuntu for every push and pull request to `main`. Jobs time out after 15
minutes, so a hung test fails the run rather than holding it for GitHub's
six-hour default.

### Benchmarks

The benchmark harness times the four Q-learning variants: whole training runs
with [hyperfine](https://github.com/sharkdp/hyperfine), and the agent and
environment steps on their own with criterion (Rust) and `timeit` (Python).
It needs hyperfine on `PATH`:

```bash
cargo install --locked hyperfine --version 2.0.0
uv run python reinforcement_learning/python/benchmark.py
```

It writes raw samples, a machine description and `summary.json` to
`results/benchmarks/<UTC time>/`, and exits with an error if any measurement's
coefficient of variation exceeds 5%.

To see where Python-driven training spends its time, record a flame graph with
py-spy, installed in `.venv` by `uv sync`. On macOS, py-spy needs `sudo`:

```bash
sudo .venv/bin/py-spy record -o profile.svg -- \
  .venv/bin/python reinforcement_learning/python/train_frozen_lake.py python 20000 0
```

## Roadmap

Each algorithm will be implemented three ways (pure Python, Rust agent with a
Gymnasium environment, and pure Rust), checked for equivalent learning
behaviour, and benchmarked before the next one starts.

- [ ] Tabular Q-learning on FrozenLake
- [ ] Further algorithms from the specialization

## License

MIT. See [LICENSE](LICENSE).
