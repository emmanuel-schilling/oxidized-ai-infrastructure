"""Trains a Q-learning agent from Python on Gymnasium's slippery FrozenLake for
end-to-end benchmarks, with the pure-Python agent or the Rust one in oxidized_rl:

    uv run python reinforcement_learning/python/train_frozen_lake.py <python|rust> <episodes> <seed>

The settings match the equivalence tests: run `seed` seeds the agent with `seed`
and the environment with 1000 + `seed` on the first reset. Prints the episode
and step counts as JSON, like the Rust example train_frozen_lake.
"""

import argparse
import json

import gymnasium as gym
import q_learning
from q_learning import QLearningParams, train_episode

import oxidized_rl

PARAMS = QLearningParams(epsilon=0.1, step_size=0.02, discount=0.9)
ENV_SEED_OFFSET = 1_000
AGENTS = {"python": q_learning.QLearningAgent, "rust": oxidized_rl.QLearningAgent}


def non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning on slippery FrozenLake.")
    parser.add_argument("agent", choices=sorted(AGENTS))
    parser.add_argument("episodes", type=non_negative)
    parser.add_argument("seed", type=non_negative)
    return parser.parse_args(argv)


def train(agent_name: str, episodes: int, seed: int) -> int:
    """Trains a fresh agent for `episodes` and returns the total number of steps."""
    env = gym.make("FrozenLake-v1", is_slippery=True)
    agent = AGENTS[agent_name](16, 4, PARAMS, seed)
    steps = 0
    for episode in range(episodes):
        summary = train_episode(agent, env, seed=ENV_SEED_OFFSET + seed if episode == 0 else None)
        steps += summary.steps
    return steps


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    steps = train(args.agent, args.episodes, args.seed)
    print(json.dumps({"episodes": args.episodes, "steps": steps}))


if __name__ == "__main__":
    main()
