"""Trains a Q-learning agent from Python on Gymnasium's slippery FrozenLake, with
the pure-Python agent or the Rust one in oxidized_rl:

    uv run python reinforcement_learning/python/train_frozen_lake.py \\
        [--agent python|rust] [--episodes N] [--seed S] [--evaluate]

Defaults: the Python agent, 20,000 episodes, seed 0. The settings match the
equivalence tests: seed S seeds the agent with S and the environment with
1000 + S on the first reset. Prints the episode and step counts as JSON, like
the Rust example train_frozen_lake, plus, with --evaluate, the trained greedy
policy's exact probability of reaching the goal within the step limit.
"""

import argparse
import json
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import q_learning
from q_learning import QLearningParams, train_episode

import oxidized_rl

PARAMS = QLearningParams(epsilon=0.1, step_size=0.02, discount=0.9)
ENV_SEED_OFFSET = 1_000
MAX_EPISODE_STEPS = 100
AGENTS = {"python": q_learning.QLearningAgent, "rust": oxidized_rl.QLearningAgent}

Transitions = dict[int, dict[int, list[tuple[float, int, float, bool]]]]
QValues = Callable[[int], list[float]]


def success_probability(q_values: QValues, transitions: Transitions) -> float:
    """The exact probability that the greedy policy reaches the goal from the
    start state within the step limit. Tied actions share the choice equally,
    as greedy_action does on average, so scoring draws nothing from the agent's
    generator and leaves training unchanged."""
    # reach[s]: probability of reaching the goal from s within k steps.
    reach = [0.0] * len(transitions)
    for _ in range(MAX_EPISODE_STEPS):
        next_reach = [0.0] * len(reach)
        for state in range(len(reach)):
            q = q_values(state)
            best = [action for action, value in enumerate(q) if value == max(q)]
            for action in best:
                for probability, next_state, reward, terminated in transitions[state][action]:
                    future = 0.0 if terminated else reach[next_state]
                    next_reach[state] += probability * (reward + future) / len(best)
        reach = next_reach
    return reach[0]


def non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning on slippery FrozenLake.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="python")
    parser.add_argument("--episodes", type=non_negative, default=20_000)
    parser.add_argument("--seed", type=non_negative, default=0)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="also print the trained greedy policy's success probability",
    )
    return parser.parse_args(argv)


def train(agent_name: str, episodes: int, seed: int) -> tuple[Any, Any, int]:
    """Trains a fresh agent for `episodes` and returns it, the environment and
    the total number of steps."""
    env = gym.make("FrozenLake-v1", is_slippery=True)
    agent = AGENTS[agent_name](16, 4, PARAMS, seed)
    steps = 0
    for episode in range(episodes):
        summary = train_episode(agent, env, seed=ENV_SEED_OFFSET + seed if episode == 0 else None)
        steps += summary.steps
    return agent, env, steps


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    agent, env, steps = train(args.agent, args.episodes, args.seed)
    result: dict[str, float] = {"episodes": args.episodes, "steps": steps}
    if args.evaluate:
        result["success_probability"] = success_probability(agent.q_values, env.unwrapped.P)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
