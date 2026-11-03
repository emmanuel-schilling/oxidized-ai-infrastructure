"""Draws the charts for a committed results folder as SVG:

    uv run python reinforcement_learning/python/plot_results.py <results folder>

The folder holds the benchmark runner's summary.json and the equivalence
tests' q_learning_equivalence_{rust,python}.json. Writes learning_curves.svg,
final_scores.svg, time_per_step.svg and step_breakdown.svg next to them.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

MAX_EPISODE_STEPS = 100

# Reference palette: chart surface, text and gridline tokens, and the first
# three categorical slots, which stay distinguishable in every pairing.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# The Rust agent learns identical curves whether Rust or Python drives it on
# Gymnasium (the equivalence tests check this exactly), so B and D share a line.
CURVE_GROUPS = [
    ("A  native Rust", ["rust_native"]),
    ("B, D  Rust agent\non Gymnasium", ["rust_gymnasium", "rust_from_python"]),
    ("C  pure Python", ["python"]),
]
TIMING_LABELS = {
    "rust_native": "A  native Rust",
    "rust_gymnasium": "B  Rust agent, Gymnasium,\n    Rust loop",
    "rust_from_python": "D  Rust agent, Gymnasium,\n    Python loop",
    "python": "C  pure Python",
}
BREAKDOWN_LABELS = {
    "native_lake/step": "Native lake step",
    "rust_agent/select_action": "Rust agent: choose action",
    "rust_agent/update": "Rust agent: update",
    "rust_agent_from_python/select_action": "Rust agent from Python: choose action",
    "rust_agent_from_python/update": "Rust agent from Python: update",
    "python_agent/select_action": "Python agent: choose action",
    "python_agent/update": "Python agent: update",
    "gymnasium_lake/step": "Gymnasium step from Python",
    "gymnasium_lake_via_pyo3/step": "Gymnasium step from Rust (PyO3)",
}

plt.rcParams.update(
    {
        "svg.fonttype": "path",  # text drawn as outlines: looks the same everywhere
        "svg.hashsalt": "oxidized",  # stable element ids, so reruns diff cleanly
        "font.size": 10,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def success_probability(transitions: Any, policy: dict[int, int] | None) -> float:
    """The probability of reaching the goal from the start within the step limit,
    following `policy` (state -> action), or the best achievable when None."""
    reach = [0.0] * len(transitions)
    for _ in range(MAX_EPISODE_STEPS):
        reach = [
            max(
                sum(
                    p * (reward + (0.0 if terminated else reach[next_state]))
                    for p, next_state, reward, terminated in outcomes
                )
                for action, outcomes in transitions[state].items()
                if policy is None or action == policy[state]
            )
            for state in range(len(reach))
        ]
    return reach[0]


def optimal_policy(transitions: Any, discount: float) -> dict[int, int]:
    """The policy maximizing the discounted return, by value iteration."""

    def action_value(state: int, action: int, values: list[float]) -> float:
        return sum(
            p * (reward + (0.0 if terminated else discount * values[next_state]))
            for p, next_state, reward, terminated in transitions[state][action]
        )

    values = [0.0] * len(transitions)
    for _ in range(2_000):
        values = [
            max(action_value(state, action, values) for action in transitions[state])
            for state in range(len(values))
        ]
    return {
        state: max(transitions[state], key=lambda action: action_value(state, action, values))
        for state in range(len(values))
    }


def slippery_lake_transitions() -> Any:
    return gym.make("FrozenLake-v1", is_slippery=True).unwrapped.P  # type: ignore[attr-defined]


def style(axes: Axes, grid_axis: Literal["x", "y"]) -> None:
    axes.grid(axis=grid_axis, color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(length=0)


def save(figure: Figure, path: Path) -> None:
    # No creation date, so an unchanged chart is byte-identical on rerun.
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def plot_learning_curves(
    curves: dict[str, list[list[float]]], checkpoint_every: int, best: float, path: Path
) -> None:
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    checkpoints = len(curves["rust_native"][0])
    episodes = [(i + 1) * checkpoint_every for i in range(checkpoints)]
    for color, (label, variants) in zip(SERIES, CURVE_GROUPS, strict=True):
        runs = curves[variants[0]]
        at = [[run[i] for run in runs] for i in range(checkpoints)]
        mean = [statistics.fmean(values) for values in at]
        error = [statistics.stdev(values) / len(values) ** 0.5 for values in at]
        low = [m - e for m, e in zip(mean, error, strict=True)]
        high = [m + e for m, e in zip(mean, error, strict=True)]
        axes.fill_between(episodes, low, high, color=color, alpha=0.18, linewidth=0)
        axes.plot(episodes, mean, color=color, linewidth=2, label=label, solid_capstyle="round")
    axes.axhline(best, color=TEXT_SECONDARY, linewidth=1)
    axes.annotate(
        f"optimal policy: {best:.2f}",
        (episodes[0], best),
        xytext=(0, 4),
        textcoords="offset points",
        color=TEXT_SECONDARY,
    )
    axes.set_xlim(0, episodes[-1] * 1.02)
    axes.set_ylim(0, 0.85)
    axes.set_xlabel("Training episodes")
    axes.set_ylabel("Greedy success probability")
    axes.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}K" if x else "0")
    style(axes, "y")
    axes.legend(loc="lower right", frameon=False)
    axes.set_title(
        "All variants learn alike: mean of 10 seeds, band ±1 standard error",
        loc="left",
        fontsize=10,
        color=TEXT_PRIMARY,
    )
    save(figure, path)


def plot_final_scores(curves: dict[str, list[list[float]]], best: float, path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    for x, (color, (_, variants)) in enumerate(zip(SERIES, CURVE_GROUPS, strict=True)):
        finals = [run[-1] for run in curves[variants[0]]]
        # Spread the dots sideways by seed so equal scores stay visible.
        offsets = [(i - (len(finals) - 1) / 2) * 0.045 for i in range(len(finals))]
        axes.scatter(
            [x + o for o in offsets], finals, s=64, color=color, edgecolors=SURFACE, linewidths=2
        )
        mean = statistics.fmean(finals)
        axes.hlines(mean, x - 0.25, x + 0.25, color=TEXT_PRIMARY, linewidth=2)
        axes.annotate(f"{mean:.2f}", (x + 0.27, mean), va="center", color=TEXT_SECONDARY)
    axes.axhline(best, color=TEXT_SECONDARY, linewidth=1)
    axes.annotate(
        f"optimal policy: {best:.2f}",
        (-0.4, best),
        xytext=(0, 4),
        textcoords="offset points",
        color=TEXT_SECONDARY,
    )
    axes.set_xticks(range(len(CURVE_GROUPS)), [label for label, _ in CURVE_GROUPS])
    axes.set_xlim(-0.55, len(CURVE_GROUPS) - 0.35)
    axes.set_ylim(0.3, 0.85)
    axes.set_ylabel("Greedy success probability")
    style(axes, "y")
    axes.set_title(
        "After 20,000 episodes: one dot per seed, black bar at the mean",
        loc="left",
        fontsize=10,
        color=TEXT_PRIMARY,
    )
    save(figure, path)


def microseconds(seconds: float) -> str:
    value = seconds * 1e6
    return f"{value:.3f} µs" if value < 0.1 else f"{value:.2f} µs"


def plot_bars(values: dict[str, float], labels: dict[str, str], title: str, path: Path) -> None:
    """Horizontal bars on a log scale, largest at the top, values at the tips."""
    ordered = sorted(values, key=values.__getitem__)
    figure, axes = plt.subplots(figsize=(8.5, 0.45 * len(ordered) + 1.2))
    positions = range(len(ordered))
    axes.barh(positions, [values[k] * 1e6 for k in ordered], height=0.5, color=SERIES[0])
    for y, key in zip(positions, ordered, strict=True):
        axes.annotate(
            microseconds(values[key]),
            (values[key] * 1e6, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
        )
    axes.set_yticks(list(positions), [labels[k] for k in ordered])
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
    low = min(values.values()) * 1e6
    high = max(values.values()) * 1e6
    axes.set_xlim(low / 3, high * 10)
    axes.set_xlabel("Microseconds (log scale)")
    style(axes, "x")
    axes.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    save(figure, path)


def summary_discount(folder: Path) -> float:
    results = json.loads((folder / "q_learning_equivalence_rust.json").read_text(encoding="utf-8"))
    return results["params"]["discount"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the charts for a results folder.")
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args(argv).folder

    summary: dict[str, Any] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    curves: dict[str, list[list[float]]] = {}
    checkpoint_every = 0
    for name in ("q_learning_equivalence_rust.json", "q_learning_equivalence_python.json"):
        results = json.loads((folder / name).read_text(encoding="utf-8"))
        curves |= results["curves"]
        checkpoint_every = results["checkpoint_every"]

    # Q-learning maximizes the discounted return, so its target is the success
    # probability of the policy that is optimal for that discount.
    discount = summary_discount(folder)
    best = success_probability(
        slippery_lake_transitions(), optimal_policy(slippery_lake_transitions(), discount)
    )
    plot_learning_curves(curves, checkpoint_every, best, folder / "learning_curves.svg")
    plot_final_scores(curves, best, folder / "final_scores.svg")
    plot_bars(
        {k: v["seconds_per_step"] for k, v in summary["training"].items()},
        TIMING_LABELS,
        "Training time per step, startup excluded",
        folder / "time_per_step.svg",
    )
    plot_bars(
        {k: v["mean"] for k, v in summary["microbenchmarks"].items()},
        BREAKDOWN_LABELS,
        "Cost of each part of a step, timed on its own",
        folder / "step_breakdown.svg",
    )


if __name__ == "__main__":
    main()
