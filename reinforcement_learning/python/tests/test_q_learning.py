"""Tests that run against both the pure-Python agent and the Rust agent in oxidized_rl.

The hand-computed cases match the Rust agent's own tests in src/agents/q_learning.rs.
"""

import json
import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import pytest
import q_learning
from q_learning import EpisodeSummary, QLearningParams, train_episode

import oxidized_rl

PARAMS = QLearningParams(epsilon=0.1, step_size=0.5, discount=0.9)
AGENT_SEED = 11
ENV_SEED = 2024
GOLDEN_Q_TABLE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "q_learning_frozen_lake_4x4_deterministic.json"
)


class AgentUnderTest(q_learning.Agent, Protocol):
    @property
    def num_states(self) -> int: ...
    @property
    def num_actions(self) -> int: ...
    def q_values(self, state: int) -> list[float]: ...
    def greedy_action(self, state: int) -> int: ...


class AgentFactory(Protocol):
    def __call__(
        self, num_states: int, num_actions: int, params: QLearningParams, seed: int
    ) -> AgentUnderTest: ...


@pytest.fixture(
    params=[q_learning.QLearningAgent, oxidized_rl.QLearningAgent], ids=["python", "rust"]
)
def agent_class(request: pytest.FixtureRequest) -> AgentFactory:
    return request.param


class ScriptedEnv:
    """Plays back fixed reset observations and step results."""

    def __init__(self, resets: list[int], steps: list[tuple[int, float, bool, bool]]) -> None:
        self.resets = deque(resets)
        self.steps = deque(steps)

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        return self.resets.popleft(), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return (*self.steps.popleft(), {})


def test_new_starts_with_a_zero_table(agent_class: AgentFactory) -> None:
    agent = agent_class(3, 2, PARAMS, seed=0)
    assert (agent.num_states, agent.num_actions) == (3, 2)
    for state in range(3):
        assert agent.q_values(state) == [0.0, 0.0]


@pytest.mark.parametrize(
    ("num_states", "num_actions", "params"),
    [
        (0, 2, PARAMS),
        (2, 0, PARAMS),
        (2, 2, replace(PARAMS, epsilon=-0.1)),
        (2, 2, replace(PARAMS, epsilon=1.1)),
        (2, 2, replace(PARAMS, epsilon=math.nan)),
        (2, 2, replace(PARAMS, step_size=0.0)),
        (2, 2, replace(PARAMS, step_size=1.5)),
        (2, 2, replace(PARAMS, discount=1.01)),
        (2, 2, replace(PARAMS, discount=math.nan)),
    ],
)
def test_invalid_configurations_are_rejected(
    agent_class: AgentFactory, num_states: int, num_actions: int, params: QLearningParams
) -> None:
    with pytest.raises(ValueError):
        agent_class(num_states, num_actions, params, seed=0)


def test_updates_match_hand_computed_values(agent_class: AgentFactory) -> None:
    # step_size 0.5, discount 0.9
    agent = agent_class(2, 2, PARAMS, seed=0)

    # Terminal: target = 1, Q(0,1) = 0 + 0.5 * (1 - 0) = 0.5
    assert agent.update(0, 1, 1.0, 1, True) == pytest.approx(1.0, abs=1e-12)
    assert agent.q_values(0)[1] == pytest.approx(0.5, abs=1e-12)

    # target = 2 + 0.9 * max(0, 0.5) = 2.45, Q(1,0) = 0.5 * 2.45 = 1.225
    assert agent.update(1, 0, 2.0, 0, False) == pytest.approx(2.45, abs=1e-12)
    assert agent.q_values(1)[0] == pytest.approx(1.225, abs=1e-12)

    # target = 0 + 0.9 * 1.225 = 1.1025, Q(0,1) = 0.5 + 0.5 * (1.1025 - 0.5) = 0.80125
    assert agent.update(0, 1, 0.0, 1, False) == pytest.approx(0.6025, abs=1e-12)
    assert agent.q_values(0)[1] == pytest.approx(0.80125, abs=1e-12)

    # Terminated ignores next_state: Q(1,1) = 0.5 * 1 = 0.5
    agent.update(1, 1, 1.0, 0, True)
    assert agent.q_values(1)[1] == pytest.approx(0.5, abs=1e-12)

    # Untouched entries stay at zero
    assert agent.q_values(0)[0] == 0.0


def test_out_of_range_states_and_actions_are_rejected(agent_class: AgentFactory) -> None:
    agent = agent_class(2, 2, PARAMS, seed=0)
    for call in [
        lambda: agent.q_values(2),
        lambda: agent.greedy_action(2),
        lambda: agent.greedy_action(-1),
        lambda: agent.select_action(2),
        lambda: agent.update(2, 0, 0.0, 0, False),
        lambda: agent.update(0, 0, 0.0, 2, False),
        lambda: agent.update(0, 2, 0.0, 0, False),
        lambda: agent.update(0, -1, 0.0, 0, False),
    ]:
        with pytest.raises(IndexError):
            call()


def test_greedy_picks_the_best_action_and_breaks_ties_uniformly(
    agent_class: AgentFactory,
) -> None:
    draws = 30_000
    agent = agent_class(2, 3, PARAMS, seed=0)
    agent.update(1, 2, 1.0, 0, True)

    counts = [0, 0, 0]
    for _ in range(draws):
        assert agent.greedy_action(1) == 2
        counts[agent.greedy_action(0)] += 1
    # 4.9 standard deviations at n = 30,000
    assert all(abs(count / draws - 1 / 3) < 0.014 for count in counts), counts


@pytest.mark.parametrize(
    ("epsilon", "expected_best"), [(0.0, 1.0), (0.3, 0.7 + 0.3 / 4), (1.0, 0.25)]
)
def test_epsilon_sets_the_exploration_rate(
    agent_class: AgentFactory, epsilon: float, expected_best: float
) -> None:
    draws = 30_000
    agent = agent_class(1, 4, replace(PARAMS, epsilon=epsilon), seed=0)
    agent.update(0, 3, 1.0, 0, True)
    best = sum(agent.select_action(0) == 3 for _ in range(draws))
    assert abs(best / draws - expected_best) < 0.014


def test_the_same_seed_gives_the_same_actions(agent_class: AgentFactory) -> None:
    def actions(seed: int) -> list[int]:
        agent = agent_class(1, 4, PARAMS, seed)
        return [agent.select_action(0) for _ in range(500)]

    assert actions(7) == actions(7)
    assert actions(7) != actions(8)


def test_episodes_bootstrap_on_truncation_but_not_on_termination(
    agent_class: AgentFactory,
) -> None:
    agent = agent_class(2, 1, PARAMS, seed=0)
    # Episode 1: state 1 -> terminal, reward 1. Episode 2: state 0 -> 1, truncated.
    env = ScriptedEnv(resets=[1, 0], steps=[(0, 1.0, True, False), (1, 0.0, False, True)])

    assert train_episode(agent, env, seed=0) == EpisodeSummary(steps=1, total_reward=1.0)
    assert agent.q_values(1)[0] == pytest.approx(0.5, abs=1e-12)
    # Terminal: Q(0) was not used as a bootstrap target, so it is still 0.
    assert agent.q_values(0)[0] == 0.0

    assert train_episode(agent, env).steps == 1
    # Truncated: 0.5 * (0 + 0.9 * 0.5) = 0.225
    assert agent.q_values(0)[0] == pytest.approx(0.225, abs=1e-12)


def test_episodes_run_until_the_environment_ends_them(agent_class: AgentFactory) -> None:
    agent = agent_class(3, 2, PARAMS, seed=0)
    env = ScriptedEnv(
        resets=[0],
        steps=[(1, 0.0, False, False), (2, 0.5, False, False), (0, 1.0, True, False)],
    )
    assert train_episode(agent, env, seed=0) == EpisodeSummary(steps=3, total_reward=1.5)
    assert not env.steps


def test_out_of_range_observations_fail_the_episode(agent_class: AgentFactory) -> None:
    agent = agent_class(2, 2, PARAMS, seed=0)
    with pytest.raises(IndexError):
        train_episode(agent, ScriptedEnv(resets=[5], steps=[]), seed=0)


def train(
    agent_class: AgentFactory, is_slippery: bool, params: QLearningParams, episodes: int
) -> tuple[AgentUnderTest, Any]:
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    agent = agent_class(16, 4, params, AGENT_SEED)
    for episode in range(episodes):
        train_episode(agent, env, seed=ENV_SEED if episode == 0 else None)
    return agent, env


def greedy_episode(agent: AgentUnderTest, env: Any) -> tuple[int, float]:
    """Runs one greedy episode and returns (steps, final reward)."""
    state, _ = env.reset()
    steps = 0
    while True:
        state, reward, terminated, truncated, _ = env.step(agent.greedy_action(int(state)))
        steps += 1
        if terminated or truncated:
            return steps, float(reward)


def test_finds_the_shortest_path_on_the_non_slippery_lake(agent_class: AgentFactory) -> None:
    agent, env = train(agent_class, is_slippery=False, params=PARAMS, episodes=500)
    # Every shortest path from 0 to 15 on the 4x4 map takes 6 steps.
    assert greedy_episode(agent, env) == (6, 1.0)


def test_beats_a_random_policy_on_the_slippery_lake(agent_class: AgentFactory) -> None:
    # Same settings as the Rust test: a step size of 0.02 averages over slips.
    agent, env = train(
        agent_class, is_slippery=True, params=replace(PARAMS, step_size=0.02), episodes=20_000
    )
    successes = sum(greedy_episode(agent, env)[1] > 0.0 for _ in range(1_000))
    # A uniform random policy succeeds about 1.5% of the time; the optimal
    # policy for discount 0.9 about 73% (value iteration on Gymnasium's table).
    assert successes / 1_000 > 0.3


def test_rust_agent_driven_from_python_matches_the_pure_rust_q_table() -> None:
    # Without slipping, Gymnasium uses no randomness after reset, so the Rust
    # agent's seeded stream fully determines training. Its table must equal the
    # one the pure-Rust run in tests/q_learning_test.rs wrote to the fixture.
    agent, _ = train(oxidized_rl.QLearningAgent, is_slippery=False, params=PARAMS, episodes=200)
    golden = json.loads(GOLDEN_Q_TABLE.read_text(encoding="utf-8"))["q_values"]
    assert [agent.q_values(state) for state in range(16)] == golden
