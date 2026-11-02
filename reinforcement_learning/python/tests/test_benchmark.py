"""Tests for the benchmark harness that need no hyperfine and no long runs."""

import json
from pathlib import Path
from typing import Any

import benchmark
import microbenchmarks
import pytest
import train_frozen_lake
from microbenchmarks import summarize


def test_summarize_reports_the_methodology_statistics() -> None:
    stats = summarize([1.0, 2.0, 3.0, 6.0])
    assert (stats.mean, stats.median, stats.min, stats.samples) == (3.0, 2.5, 1.0, 4)
    assert stats.std_dev == pytest.approx(2.1602469)
    assert stats.coefficient_of_variation == pytest.approx(0.7200823)


def test_summarize_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        summarize([1.0])


def test_too_noisy_flags_spread_above_five_percent_except_startup() -> None:
    steady = summarize([1.0, 1.01, 0.99])
    noisy = summarize([1.0, 1.2, 0.8])
    stats = {"a/train": steady, "b/train": noisy, "micro/step": noisy, "b/startup": noisy}
    assert benchmark.too_noisy(stats) == ["b/train", "micro/step"]


def test_rounds_rotate_the_order() -> None:
    names = ["a", "b", "c"]
    assert [benchmark.rotated(names, i) for i in range(4)] == [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["a", "b", "c"],
    ]


def test_commands_cover_every_variant_with_a_startup_baseline() -> None:
    episodes = {variant: 500 + i for i, variant in enumerate(benchmark.VARIANTS)}
    commands = benchmark.commands(Path("train_frozen_lake"), episodes, seed=3)
    assert len(commands) == 2 * len(benchmark.VARIANTS)
    for variant in benchmark.VARIANTS:
        assert commands[f"{variant}/train"][-2:] == [str(episodes[variant]), "3"]
        assert commands[f"{variant}/startup"][-2:] == ["0", "3"]
        assert commands[f"{variant}/train"][:-2] == commands[f"{variant}/startup"][:-2]


def hyperfine_export(times: dict[str, list[float]], unit: str = "second") -> dict[str, Any]:
    """A minimal hyperfine 2 JSON export (schema 2) with the given wall-clock times."""
    return {
        "schema_version": 2,
        "results": [
            {
                "name": name,
                "measurements": [{"time_wall_clock": {"value": t, "unit": unit}} for t in values],
            }
            for name, values in times.items()
        ],
    }


def test_samples_pool_across_rounds() -> None:
    rounds = [
        hyperfine_export({"a": [1.0, 2.0], "b": [5.0]}),
        hyperfine_export({"b": [6.0], "a": [3.0]}),
    ]
    assert benchmark.merge_samples(rounds) == {"a": [1.0, 2.0, 3.0], "b": [5.0, 6.0]}


@pytest.mark.parametrize(
    "exported",
    [
        {"results": [{"command": "a", "times": [1.0]}]},  # hyperfine 1 format
        hyperfine_export({"a": [1.0]}, unit="millisecond"),
    ],
)
def test_unexpected_hyperfine_exports_are_rejected(exported: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        benchmark.merge_samples([exported])


def test_training_time_subtracts_startup() -> None:
    samples = {}
    steps = {}
    for variant in benchmark.VARIANTS:
        samples[f"{variant}/train"] = [2.5, 2.5]
        samples[f"{variant}/startup"] = [0.5, 0.5]
        steps[f"{variant}/train"] = 1_000
    samples["python/train"] = [2.0, 3.0]  # standard error 0.5
    samples["python/startup"] = [0.25, 0.75]  # standard error 0.25
    summary = benchmark.training_summary(samples, steps)["python"]
    assert summary["steps"] == 1_000
    assert summary["training_seconds"] == pytest.approx(2.0)
    assert summary["seconds_per_step"] == pytest.approx(0.002)
    # The two standard errors add in quadrature: sqrt(0.5^2 + 0.25^2).
    assert summary["training_seconds_standard_error"] == pytest.approx(0.5590170)
    assert summary["seconds_per_step_standard_error"] == pytest.approx(0.5590170e-3)


def test_native_rust_trains_longer_by_default() -> None:
    args = benchmark.parse_args([])
    assert (args.episodes, args.native_episodes) == (20_000, 1_000_000)


def test_criterion_samples_become_seconds_per_iteration(tmp_path: Path) -> None:
    new = tmp_path / "rust_agent" / "update" / "new"
    new.mkdir(parents=True)
    (new / "benchmark.json").write_text(json.dumps({"full_id": "rust_agent/update"}))
    # 10 iterations in 200 ns, then 20 in 600 ns: 20 and 30 ns per iteration.
    (new / "sample.json").write_text(json.dumps({"iters": [10.0, 20.0], "times": [200.0, 600.0]}))
    stats = benchmark.read_criterion(tmp_path)["rust_agent/update"]
    assert stats.mean == pytest.approx(25e-9)
    assert stats.min == pytest.approx(20e-9)


def test_machine_description_names_the_toolchain() -> None:
    machine = benchmark.machine_description()
    assert machine["logical_cores"] >= 1
    assert machine["python"].startswith("3.11")
    assert machine["rustc"].startswith("rustc ")


def test_child_processes_can_find_the_virtual_environment() -> None:
    env = benchmark.run_env()
    assert env["PYO3_PYTHON"]
    assert "site-packages" in env["PYTHONPATH"]


@pytest.mark.parametrize(
    "agent_class",
    [microbenchmarks.q_learning.QLearningAgent, microbenchmarks.oxidized_rl.QLearningAgent],
)
def test_action_choice_is_timed_without_ties(agent_class: Any) -> None:
    # Ties take a slower path that training rarely hits once it is under way.
    agent = microbenchmarks.untied_agent(agent_class)
    for state in range(16):
        values = agent.q_values(state)
        assert values.count(max(values)) == 1


def test_microbenchmark_cases_run() -> None:
    cases = {
        **microbenchmarks.agent_cases("python_agent", microbenchmarks.q_learning.QLearningAgent),
        **microbenchmarks.agent_cases("rust", microbenchmarks.oxidized_rl.QLearningAgent),
        **microbenchmarks.gymnasium_step_case(),
    }
    # More calls than an episode's step limit, so the reset path runs too.
    for call in cases.values():
        for _ in range(150):
            call()
    stats = microbenchmarks.time_per_call(lambda: None, repeats=2)
    assert stats.samples == 2 and stats.mean > 0


@pytest.mark.parametrize("agent", ["python", "rust"])
def test_training_program_prints_the_step_count(
    agent: str, capsys: pytest.CaptureFixture[str]
) -> None:
    train_frozen_lake.main([agent, "50", "0"])
    printed = json.loads(capsys.readouterr().out)
    # Each episode takes between 1 and 100 steps.
    assert printed["episodes"] == 50 and 50 <= printed["steps"] <= 5_000


def test_both_python_driven_programs_take_the_same_steps() -> None:
    # The Rust agent driven from Python reproduces the Rust-driven run exactly
    # (see the equivalence tests); here it must at least match itself.
    assert train_frozen_lake.train("rust", 50, 0) == train_frozen_lake.train("rust", 50, 0)
    assert train_frozen_lake.train("rust", 0, 0) == 0


@pytest.mark.parametrize("argv", [[], ["python", "10"], ["java", "10", "0"], ["rust", "-1", "0"]])
def test_training_program_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        train_frozen_lake.parse_args(argv)
