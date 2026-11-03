"""Tests for the chart script: the exact success calculations and a full run."""

import json
from pathlib import Path

import gymnasium as gym
import plot_results
import pytest
from benchmark import VARIANTS


def lake(is_slippery: bool) -> object:
    return gym.make("FrozenLake-v1", is_slippery=is_slippery).unwrapped.P  # type: ignore[attr-defined]


def test_the_non_slippery_lake_can_always_be_crossed() -> None:
    transitions = lake(is_slippery=False)
    assert plot_results.success_probability(transitions, None) == 1.0
    policy = plot_results.optimal_policy(transitions, discount=0.9)
    assert plot_results.success_probability(transitions, policy) == 1.0


def test_discounting_trades_a_little_success_for_shorter_paths() -> None:
    transitions = lake(is_slippery=True)
    ceiling = plot_results.success_probability(transitions, None)
    discounted = plot_results.success_probability(
        transitions, plot_results.optimal_policy(transitions, discount=0.9)
    )
    # The best discounted policy matches the best seeds' greedy policies.
    assert discounted == pytest.approx(0.7298, abs=1e-4)
    assert ceiling == pytest.approx(0.7442, abs=1e-4)


def test_a_policy_that_walks_into_a_hole_never_succeeds() -> None:
    # On the non-slippery lake: down from the start to state 4, then right
    # into the hole at state 5.
    policy = {state: 1 for state in range(16)} | {4: 2}
    assert plot_results.success_probability(lake(is_slippery=False), policy) == 0.0


def write_results(folder: Path) -> None:
    curve = [0.1 * (i + 1) / 2 for i in range(4)]
    for name, variants in [
        ("q_learning_equivalence_rust.json", ["rust_native", "rust_gymnasium"]),
        ("q_learning_equivalence_python.json", ["python", "rust_from_python"]),
    ]:
        results = {
            "params": {"discount": 0.9},
            "checkpoint_every": 1_000,
            "curves": {v: [curve, [x + 0.01 for x in curve]] for v in variants},
        }
        (folder / name).write_text(json.dumps(results))
    summary = {
        "training": {v: {"seconds_per_step": 1e-6 * (i + 1)} for i, v in enumerate(VARIANTS)},
        "microbenchmarks": {
            name: {"mean": 1e-8 * (i + 1)} for i, name in enumerate(plot_results.BREAKDOWN_LABELS)
        },
    }
    (folder / "summary.json").write_text(json.dumps(summary))


def test_charts_are_drawn_and_redraw_identically(tmp_path: Path) -> None:
    write_results(tmp_path)
    plot_results.main([str(tmp_path)])
    names = ["learning_curves", "final_scores", "time_per_step", "step_breakdown"]
    first = {name: (tmp_path / f"{name}.svg").read_bytes() for name in names}
    assert all(content.startswith(b"<?xml") for content in first.values())

    plot_results.main([str(tmp_path)])
    assert {name: (tmp_path / f"{name}.svg").read_bytes() for name in names} == first
