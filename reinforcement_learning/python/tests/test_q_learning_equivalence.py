"""Statistical equivalence of the pure-Python agent and the Rust agent driven from Python.

The agents draw from different generators (NumPy's and ChaCha8), so individual
runs differ, but their mean learning curves must agree at every checkpoint. The Rust test
tests/q_learning_equivalence_test.rs compares the Rust agent on the native and
Gymnasium lakes the same way.

The comparison is slow, so opt-in: `uv run pytest -m slow`. It writes the
curves to results/q_learning_equivalence_python.json.
"""

import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest
import q_learning
from q_learning import QLearningParams, train_episode
from train_frozen_lake import Transitions, success_probability

import oxidized_rl

PARAMS = QLearningParams(epsilon=0.1, step_size=0.02, discount=0.9)
EPISODES = 20_000
CHECKPOINT_EVERY = 1_000
RUNS = 10
# Run i seeds the agent with i and the environment with ENV_SEED_OFFSET + i.
ENV_SEED_OFFSET = 1_000
# Means may differ by at most this many standard errors of their difference.
# For correct code, all 20 checkpoints pass with about 99.8% probability
# (Student's t with about 18 degrees of freedom).
MAX_STANDARD_ERRORS = 5.0
RESULTS = Path(__file__).resolve().parents[3] / "results"


def make_lake(is_slippery: bool) -> tuple[Any, Transitions]:
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    return env, env.unwrapped.P  # type: ignore[attr-defined]


def learning_curve(agent_class: Any, run: int) -> list[float]:
    """Trains a fresh agent and scores it every CHECKPOINT_EVERY episodes."""
    env, transitions = make_lake(is_slippery=True)
    agent = agent_class(16, 4, PARAMS, run)
    curve = []
    for episode in range(EPISODES):
        train_episode(agent, env, seed=ENV_SEED_OFFSET + run if episode == 0 else None)
        if (episode + 1) % CHECKPOINT_EVERY == 0:
            curve.append(success_probability(agent.q_values, transitions))
    return curve


def mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def assert_curves_agree(a: list[list[float]], b: list[list[float]]) -> None:
    """Requires the mean curves to agree within MAX_STANDARD_ERRORS at every
    checkpoint. Comparing whole curves, not a summary such as their mean,
    catches agents that learn at a different pace but end up similar."""
    for checkpoint in range(EPISODES // CHECKPOINT_EVERY):
        mean_a, se_a = mean_and_standard_error([curve[checkpoint] for curve in a])
        mean_b, se_b = mean_and_standard_error([curve[checkpoint] for curve in b])
        tolerance = MAX_STANDARD_ERRORS * math.hypot(se_a, se_b)
        assert abs(mean_a - mean_b) <= tolerance, (
            f"After {(checkpoint + 1) * CHECKPOINT_EVERY} episodes: Python {mean_a:.4f} "
            f"vs Rust {mean_b:.4f}, tolerance {tolerance:.4f}"
        )


def test_success_probability_is_one_for_the_shortest_path() -> None:
    _, transitions = make_lake(is_slippery=False)
    # Down, down, right, down, right, right: one of the 6-step shortest paths.
    path = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}

    def q_values(state: int) -> list[float]:
        return [1.0 if action == path.get(state) else 0.0 for action in range(4)]

    assert success_probability(q_values, transitions) == 1.0


def test_success_probability_of_an_untrained_agent_is_the_random_policys() -> None:
    # An all-zero table ties every action, so the policy is uniformly random,
    # which reaches the goal of the slippery lake about 1.5% of the time.
    _, transitions = make_lake(is_slippery=True)
    assert 0.005 < success_probability(lambda _: [0.0] * 4, transitions) < 0.05


@pytest.mark.slow
def test_python_and_rust_agents_give_equivalent_learning_curves() -> None:
    python = [learning_curve(q_learning.QLearningAgent, run) for run in range(RUNS)]
    rust = [learning_curve(oxidized_rl.QLearningAgent, run) for run in range(RUNS)]

    RESULTS.mkdir(exist_ok=True)
    results = {
        "description": (
            "Exact greedy success probability within 100 steps on Gymnasium's "
            "slippery FrozenLake 4x4, scored every checkpoint_every episodes. Run i "
            "seeds the agent with i and the environment with 1000 + i on the first reset."
        ),
        "params": {
            "epsilon": PARAMS.epsilon,
            "step_size": PARAMS.step_size,
            "discount": PARAMS.discount,
        },
        "episodes": EPISODES,
        "checkpoint_every": CHECKPOINT_EVERY,
        "curves": {"python": python, "rust_from_python": rust},
    }
    (RESULTS / "q_learning_equivalence_python.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    assert_curves_agree(python, rust)
