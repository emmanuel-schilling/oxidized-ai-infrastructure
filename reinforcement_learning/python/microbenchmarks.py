"""Microbenchmarks of the per-step work as Python sees it, the counterpart of the
criterion benchmarks in reinforcement_learning/benches/.

Each case repeats `timeit` runs and reports seconds per call. The Rust agent's
cases include the cost of calling into the extension module.
"""

import math
import statistics
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import q_learning
from q_learning import QLearningParams

import oxidized_rl

PARAMS = QLearningParams(epsilon=0.1, step_size=0.02, discount=0.9)


@dataclass(frozen=True)
class Stats:
    """Summary of repeated timings, in seconds."""

    mean: float
    std_dev: float
    median: float
    min: float
    samples: int

    @property
    def coefficient_of_variation(self) -> float:
        return self.std_dev / self.mean if self.mean > 0 else math.inf

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "coefficient_of_variation": self.coefficient_of_variation}


def summarize(samples: list[float]) -> Stats:
    if len(samples) < 2:
        raise ValueError("need at least two samples")
    return Stats(
        mean=statistics.fmean(samples),
        std_dev=statistics.stdev(samples),
        median=statistics.median(samples),
        min=min(samples),
        samples=len(samples),
    )


def time_per_call(call: Callable[[], object], repeats: int) -> Stats:
    """Times `call` in `repeats` batches, each long enough for a stable clock
    reading, and returns seconds per call."""
    timer = timeit.Timer(call)
    number, _ = timer.autorange()
    return summarize([total / number for total in timer.repeat(repeat=repeats, number=number)])


def untied_agent(agent_class: Any) -> Any:
    """An agent with one best action per state, as after early training. On an
    all-zero table every action choice would tie and take the slower
    tie-breaking path."""
    agent = agent_class(16, 4, PARAMS, 0)
    for state in range(16):
        agent.update(state, state % 4, 1.0, 0, True)
    return agent


def agent_cases(name: str, agent_class: Any) -> dict[str, Callable[[], object]]:
    select_agent = untied_agent(agent_class)
    update_agent = agent_class(16, 4, PARAMS, 0)
    counter = {"state": 0, "step": 0}

    def select_action() -> object:
        counter["state"] = (counter["state"] + 1) % 16
        return select_agent.select_action(counter["state"])

    def update() -> object:
        counter["step"] = (counter["step"] + 1) % 64
        state, action = divmod(counter["step"], 4)
        return update_agent.update(state, action, 0.0, (state + 1) % 16, False)

    return {f"{name}/select_action": select_action, f"{name}/update": update}


def gymnasium_step_case() -> dict[str, Callable[[], object]]:
    # Episodes that end are reset inside the timed call, as in training.
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env.reset(seed=0)
    counter = {"action": 0}

    def step() -> object:
        counter["action"] = (counter["action"] + 1) % 4
        _, _, terminated, truncated, _ = env.step(counter["action"])
        if terminated or truncated:
            env.reset()
        return None

    return {"gymnasium_lake/step": step}


def run(repeats: int = 30) -> dict[str, Stats]:
    cases = {
        **agent_cases("python_agent", q_learning.QLearningAgent),
        **agent_cases("rust_agent_from_python", oxidized_rl.QLearningAgent),
        **gymnasium_step_case(),
    }
    return {name: time_per_call(call, repeats) for name, call in cases.items()}
