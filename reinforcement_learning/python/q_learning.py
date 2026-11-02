"""Tabular Q-learning in pure Python, mirroring the Rust agent in src/agents/q_learning.rs.

Written plainly in the style of the University of Alberta RL specialization:
a NumPy Q-table, epsilon-greedy exploration, and greedy ties broken uniformly
at random.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class QLearningParams:
    """Hyperparameters for tabular Q-learning."""

    epsilon: float
    """Probability of taking a uniformly random action, in [0, 1]"""
    step_size: float
    """Learning rate (alpha), in (0, 1]"""
    discount: float
    """Discount factor (gamma), in [0, 1]"""


@dataclass(frozen=True)
class EpisodeSummary:
    """What happened during one training episode."""

    steps: int
    total_reward: float


class DiscreteEnv(Protocol):
    """The Gymnasium API subset training uses, with integer states and actions."""

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]: ...


class Agent(Protocol):
    """What training needs from an agent: this module's QLearningAgent or the
    Rust one in oxidized_rl."""

    def select_action(self, state: int) -> int: ...

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> float: ...


class QLearningAgent:
    def __init__(
        self, num_states: int, num_actions: int, params: QLearningParams, seed: int
    ) -> None:
        if num_states <= 0 or num_actions <= 0:
            raise ValueError("the Q-table needs at least one state and one action")
        # The comparisons are False for NaN, so NaN is rejected too.
        if not 0.0 <= params.epsilon <= 1.0:
            raise ValueError(f"epsilon {params.epsilon} is outside [0, 1]")
        if not 0.0 < params.step_size <= 1.0:
            raise ValueError(f"step_size {params.step_size} is outside (0, 1]")
        if not 0.0 <= params.discount <= 1.0:
            raise ValueError(f"discount {params.discount} is outside [0, 1]")

        self.num_states = num_states
        self.num_actions = num_actions
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((num_states, num_actions))

    def _check_state(self, state: int) -> None:
        if not 0 <= state < self.num_states:
            raise IndexError(f"State {state} is outside the Q-table")

    def q_values(self, state: int) -> list[float]:
        """The action values for `state`, indexed by action."""
        self._check_state(state)
        return self.q[state].tolist()

    def greedy_action(self, state: int) -> int:
        """The highest-valued action in `state`, with ties broken uniformly at random."""
        self._check_state(state)
        values = self.q[state]
        ties = np.flatnonzero(values == values.max())
        # Draw only when there is a choice to make.
        if len(ties) == 1:
            return int(ties[0])
        return int(self.rng.choice(ties))

    def select_action(self, state: int) -> int:
        """With probability epsilon a uniformly random action, otherwise the greedy one."""
        self._check_state(state)
        if self.rng.random() < self.params.epsilon:
            return int(self.rng.integers(self.num_actions))
        return self.greedy_action(state)

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> float:
        """Applies one Q-learning update and returns the TD error.

        The bootstrap term is dropped only when the episode terminated. A
        truncated episode, cut off by a step limit, still bootstraps from
        `next_state`.
        """
        if not 0 <= action < self.num_actions:
            raise IndexError(f"Action {action} is outside the Q-table")
        self._check_state(next_state)
        self._check_state(state)

        if terminated:
            target = reward
        else:
            target = reward + self.params.discount * float(self.q[next_state].max())
        td_error = target - float(self.q[state, action])
        self.q[state, action] += self.params.step_size * td_error
        return td_error


def train_episode(agent: Agent, env: DiscreteEnv, seed: int | None = None) -> EpisodeSummary:
    """Runs one episode on `env`, with `agent` learning from every step, until it
    terminates or is truncated. The environment must end episodes itself, for
    example through a step limit.

    `seed` is passed to `reset`; pass None after the first episode to continue
    the environment's random stream.
    """
    state, _ = env.reset(seed=seed)
    state = int(state)
    steps = 0
    total_reward = 0.0
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state, reward = int(next_state), float(reward)
        agent.update(state, action, reward, next_state, terminated)

        steps += 1
        total_reward += reward
        if terminated or truncated:
            return EpisodeSummary(steps, total_reward)
        state = next_state
