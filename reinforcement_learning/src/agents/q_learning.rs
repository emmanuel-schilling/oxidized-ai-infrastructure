// reinforcement_learning/src/agents/q_learning.rs

use crate::core::environment::Environment;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fmt;

/// Hyperparameters for tabular Q-learning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QLearningParams {
    /// Probability of taking a uniformly random action, in [0, 1]
    pub epsilon: f64,
    /// Learning rate (alpha), in (0, 1]
    pub step_size: f64,
    /// Discount factor (gamma), in [0, 1]
    pub discount: f64,
}

/// Error type for invalid agent configurations and out-of-range states or actions
#[derive(Debug, Clone, PartialEq)]
pub enum QLearningError {
    InvalidParameter(String),
    InvalidState(usize),
    InvalidAction(usize),
}

impl fmt::Display for QLearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid Q-learning parameter: {msg}"),
            Self::InvalidState(state) => write!(f, "State {state} is outside the Q-table"),
            Self::InvalidAction(action) => write!(f, "Action {action} is outside the Q-table"),
        }
    }
}

impl std::error::Error for QLearningError {}

/// Error from a training episode: either the agent or the environment failed.
#[derive(Debug)]
pub enum EpisodeError<E> {
    Agent(QLearningError),
    Environment(E),
}

impl<E: fmt::Display> fmt::Display for EpisodeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agent(err) => write!(f, "Agent error: {err}"),
            Self::Environment(err) => write!(f, "Environment error: {err}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for EpisodeError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Agent(err) => Some(err),
            Self::Environment(err) => Some(err),
        }
    }
}

/// What happened during one training episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpisodeSummary {
    pub steps: usize,
    pub total_reward: f64,
}

/// A tabular Q-learning agent with epsilon-greedy exploration, following the
/// University of Alberta RL specialization: greedy ties are broken uniformly
/// at random, and Q-values start at zero.
pub struct QLearning {
    /// Row-major `num_states x num_actions` table
    q: Vec<f64>,
    num_actions: usize,
    params: QLearningParams,
    rng: ChaCha8Rng,
}

impl QLearning {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        params: QLearningParams,
        seed: u64,
    ) -> Result<Self, QLearningError> {
        if num_states == 0 || num_actions == 0 {
            return Err(QLearningError::InvalidParameter(
                "the Q-table needs at least one state and one action".into(),
            ));
        }
        let size = num_states
            .checked_mul(num_actions)
            .ok_or_else(|| QLearningError::InvalidParameter("the Q-table is too large".into()))?;
        // The range checks also reject NaN.
        if !(0.0..=1.0).contains(&params.epsilon) {
            return Err(QLearningError::InvalidParameter(format!(
                "epsilon {} is outside [0, 1]",
                params.epsilon
            )));
        }
        if !(params.step_size > 0.0 && params.step_size <= 1.0) {
            return Err(QLearningError::InvalidParameter(format!(
                "step_size {} is outside (0, 1]",
                params.step_size
            )));
        }
        if !(0.0..=1.0).contains(&params.discount) {
            return Err(QLearningError::InvalidParameter(format!(
                "discount {} is outside [0, 1]",
                params.discount
            )));
        }

        Ok(Self {
            q: vec![0.0; size],
            num_actions,
            params,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    pub fn num_states(&self) -> usize {
        self.q.len() / self.num_actions
    }

    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Index of `state`'s first entry in the table.
    fn row_start(&self, state: usize) -> Result<usize, QLearningError> {
        state
            .checked_mul(self.num_actions)
            .filter(|&start| start < self.q.len())
            .ok_or(QLearningError::InvalidState(state))
    }

    /// The action values for `state`, indexed by action.
    pub fn q_values(&self, state: usize) -> Result<&[f64], QLearningError> {
        let start = self.row_start(state)?;
        Ok(&self.q[start..start + self.num_actions])
    }

    /// The highest-valued action in `state`, with ties broken uniformly at random.
    pub fn greedy_action(&mut self, state: usize) -> Result<usize, QLearningError> {
        let start = self.row_start(state)?;
        let values = &self.q[start..start + self.num_actions];

        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ties = values.iter().filter(|&&value| value == max).count();
        // Draw only when there is a choice to make.
        let pick = if ties > 1 {
            self.rng.gen_range(0..ties)
        } else {
            0
        };
        Ok(values
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value == max)
            .nth(pick)
            .map_or(0, |(action, _)| action))
    }

    /// The greedy policy as (action, probability) pairs for each state: tied
    /// highest-valued actions share the probability equally, which is the
    /// distribution `greedy_action` samples from. Reading it draws nothing from
    /// the agent's generator.
    pub fn greedy_policy(&self) -> Vec<Vec<(usize, f64)>> {
        self.q
            .chunks(self.num_actions)
            .map(|values| {
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let best: Vec<usize> = (0..values.len()).filter(|&a| values[a] == max).collect();
                let probability = 1.0 / best.len() as f64;
                best.into_iter().map(|a| (a, probability)).collect()
            })
            .collect()
    }

    /// With probability epsilon a uniformly random action, otherwise the greedy one.
    pub fn select_action(&mut self, state: usize) -> Result<usize, QLearningError> {
        if state >= self.num_states() {
            return Err(QLearningError::InvalidState(state));
        }
        let draw: f64 = self.rng.sample(Standard);
        if draw < self.params.epsilon {
            Ok(self.rng.gen_range(0..self.num_actions))
        } else {
            self.greedy_action(state)
        }
    }

    /// Applies one Q-learning update and returns the TD error:
    ///
    /// `Q(s, a) += step_size * (reward + discount * max_a' Q(s', a') - Q(s, a))`
    ///
    /// The bootstrap term is dropped only when the episode `terminated`. A
    /// truncated episode, cut off by a step limit, still bootstraps from `next_state`.
    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        terminated: bool,
    ) -> Result<f64, QLearningError> {
        if action >= self.num_actions {
            return Err(QLearningError::InvalidAction(action));
        }
        let next_max = self
            .q_values(next_state)?
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let index = self.row_start(state)? + action;

        let target = if terminated {
            reward
        } else {
            reward + self.params.discount * next_max
        };
        let value = &mut self.q[index];
        let td_error = target - *value;
        *value += self.params.step_size * td_error;
        Ok(td_error)
    }

    /// Runs one episode on `env`, learning from every step, until it terminates
    /// or is truncated. The environment must end episodes itself, for example
    /// through a step limit.
    ///
    /// `seed` is passed to `reset`; pass `None` after the first episode to
    /// continue the environment's random stream.
    pub fn train_episode<E>(
        &mut self,
        env: &mut E,
        seed: Option<u64>,
    ) -> Result<EpisodeSummary, EpisodeError<E::Error>>
    where
        E: Environment<Observation = usize, Action = usize>,
    {
        let mut state = env.reset(seed).map_err(EpisodeError::Environment)?;
        let mut summary = EpisodeSummary {
            steps: 0,
            total_reward: 0.0,
        };
        loop {
            let action = self.select_action(state).map_err(EpisodeError::Agent)?;
            let (next_state, reward, terminated, truncated) =
                env.step(action).map_err(EpisodeError::Environment)?;
            self.update(state, action, reward, next_state, terminated)
                .map_err(EpisodeError::Agent)?;

            summary.steps += 1;
            summary.total_reward += reward;
            if terminated || truncated {
                return Ok(summary);
            }
            state = next_state;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::convert::Infallible;

    const PARAMS: QLearningParams = QLearningParams {
        epsilon: 0.1,
        step_size: 0.5,
        discount: 0.9,
    };

    fn agent(num_states: usize, num_actions: usize, params: QLearningParams) -> QLearning {
        QLearning::new(num_states, num_actions, params, 0).unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    /// Plays back fixed reset observations and step results.
    struct ScriptedEnv {
        resets: VecDeque<usize>,
        steps: VecDeque<(usize, f64, bool, bool)>,
    }

    impl Environment for ScriptedEnv {
        type Observation = usize;
        type Action = usize;
        type Error = Infallible;

        fn reset(&mut self, _seed: Option<u64>) -> Result<usize, Infallible> {
            Ok(self.resets.pop_front().unwrap())
        }

        fn step(&mut self, _action: usize) -> Result<(usize, f64, bool, bool), Infallible> {
            Ok(self.steps.pop_front().unwrap())
        }
    }

    #[test]
    fn new_starts_with_a_zero_table() {
        let agent = agent(3, 2, PARAMS);
        assert_eq!(agent.num_states(), 3);
        assert_eq!(agent.num_actions(), 2);
        for state in 0..3 {
            assert_eq!(agent.q_values(state).unwrap(), [0.0, 0.0]);
        }
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let with = |epsilon, step_size, discount| QLearningParams {
            epsilon,
            step_size,
            discount,
        };
        let cases = [
            (0, 2, PARAMS),
            (2, 0, PARAMS),
            (usize::MAX, 2, PARAMS),
            (2, 2, with(-0.1, 0.5, 0.9)),
            (2, 2, with(1.1, 0.5, 0.9)),
            (2, 2, with(f64::NAN, 0.5, 0.9)),
            (2, 2, with(0.1, 0.0, 0.9)),
            (2, 2, with(0.1, 1.5, 0.9)),
            (2, 2, with(0.1, 0.5, 1.01)),
            (2, 2, with(0.1, 0.5, f64::NAN)),
        ];
        for (num_states, num_actions, params) in cases {
            assert!(
                matches!(
                    QLearning::new(num_states, num_actions, params, 0),
                    Err(QLearningError::InvalidParameter(_))
                ),
                "{num_states} x {num_actions}, {params:?} should be rejected"
            );
        }
    }

    #[test]
    fn updates_match_hand_computed_values() {
        // step_size 0.5, discount 0.9
        let mut agent = agent(2, 2, PARAMS);

        // Terminal: target = 1, Q(0,1) = 0 + 0.5 * (1 - 0) = 0.5
        assert_close(agent.update(0, 1, 1.0, 1, true).unwrap(), 1.0);
        assert_close(agent.q_values(0).unwrap()[1], 0.5);

        // target = 2 + 0.9 * max(0, 0.5) = 2.45, Q(1,0) = 0.5 * 2.45 = 1.225
        assert_close(agent.update(1, 0, 2.0, 0, false).unwrap(), 2.45);
        assert_close(agent.q_values(1).unwrap()[0], 1.225);

        // target = 0 + 0.9 * 1.225 = 1.1025, Q(0,1) = 0.5 + 0.5 * (1.1025 - 0.5) = 0.80125
        assert_close(agent.update(0, 1, 0.0, 1, false).unwrap(), 0.6025);
        assert_close(agent.q_values(0).unwrap()[1], 0.80125);

        // Terminated ignores next_state: Q(1,1) = 0.5 * 1 = 0.5
        agent.update(1, 1, 1.0, 0, true).unwrap();
        assert_close(agent.q_values(1).unwrap()[1], 0.5);

        // Untouched entries stay at zero
        assert_eq!(agent.q_values(0).unwrap()[0], 0.0);
    }

    #[test]
    fn out_of_range_states_and_actions_are_rejected() {
        let mut agent = agent(2, 2, PARAMS);
        assert_eq!(agent.q_values(2), Err(QLearningError::InvalidState(2)));
        assert_eq!(
            agent.q_values(usize::MAX),
            Err(QLearningError::InvalidState(usize::MAX))
        );
        assert_eq!(agent.greedy_action(2), Err(QLearningError::InvalidState(2)));
        assert_eq!(agent.select_action(2), Err(QLearningError::InvalidState(2)));
        assert_eq!(
            agent.update(2, 0, 0.0, 0, false),
            Err(QLearningError::InvalidState(2))
        );
        assert_eq!(
            agent.update(0, 0, 0.0, 2, false),
            Err(QLearningError::InvalidState(2))
        );
        assert_eq!(
            agent.update(0, 2, 0.0, 0, false),
            Err(QLearningError::InvalidAction(2))
        );
    }

    #[test]
    fn greedy_picks_the_best_action_and_breaks_ties_uniformly() {
        const DRAWS: usize = 30_000;
        let mut agent = agent(2, 3, PARAMS);
        agent.update(1, 2, 1.0, 0, true).unwrap();

        let mut counts = [0usize; 3];
        for _ in 0..DRAWS {
            assert_eq!(agent.greedy_action(1), Ok(2));
            counts[agent.greedy_action(0).unwrap()] += 1;
        }
        for count in counts {
            let frequency = count as f64 / DRAWS as f64;
            // 4.9 standard deviations at n = 30,000
            assert!((frequency - 1.0 / 3.0).abs() < 0.014, "counts {counts:?}");
        }
    }

    #[test]
    fn epsilon_sets_the_exploration_rate() {
        const DRAWS: usize = 30_000;
        for (epsilon, expected_best) in [(0.0, 1.0), (0.3, 0.7 + 0.3 / 4.0), (1.0, 0.25)] {
            let mut agent = agent(1, 4, QLearningParams { epsilon, ..PARAMS });
            agent.update(0, 3, 1.0, 0, true).unwrap();
            let best = (0..DRAWS)
                .filter(|_| agent.select_action(0) == Ok(3))
                .count();
            let frequency = best as f64 / DRAWS as f64;
            assert!(
                (frequency - expected_best).abs() < 0.014,
                "epsilon {epsilon}: best action frequency {frequency}"
            );
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_actions() {
        let actions = |seed| {
            let mut agent = QLearning::new(1, 4, PARAMS, seed).unwrap();
            (0..500)
                .map(|_| agent.select_action(0).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(actions(7), actions(7));
        assert_ne!(actions(7), actions(8));
    }

    #[test]
    fn episodes_bootstrap_on_truncation_but_not_on_termination() {
        let mut agent = agent(2, 1, PARAMS);
        let mut env = ScriptedEnv {
            resets: VecDeque::from([1, 0]),
            // Episode 1: state 1 -> terminal, reward 1. Episode 2: state 0 -> 1, truncated.
            steps: VecDeque::from([(0, 1.0, true, false), (1, 0.0, false, true)]),
        };

        let first = agent.train_episode(&mut env, Some(0)).unwrap();
        assert_eq!(
            first,
            EpisodeSummary {
                steps: 1,
                total_reward: 1.0
            }
        );
        assert_close(agent.q_values(1).unwrap()[0], 0.5);
        // Terminal: Q(0) was not used as a bootstrap target, so it is still 0.
        assert_eq!(agent.q_values(0).unwrap()[0], 0.0);

        let second = agent.train_episode(&mut env, None).unwrap();
        assert_eq!(second.steps, 1);
        // Truncated: 0.5 * (0 + 0.9 * 0.5) = 0.225
        assert_close(agent.q_values(0).unwrap()[0], 0.225);
    }

    #[test]
    fn episodes_run_until_the_environment_ends_them() {
        let mut agent = agent(3, 2, PARAMS);
        let mut env = ScriptedEnv {
            resets: VecDeque::from([0]),
            steps: VecDeque::from([
                (1, 0.0, false, false),
                (2, 0.5, false, false),
                (0, 1.0, true, false),
            ]),
        };
        let summary = agent.train_episode(&mut env, Some(0)).unwrap();
        assert_eq!(
            summary,
            EpisodeSummary {
                steps: 3,
                total_reward: 1.5
            }
        );
        assert!(env.steps.is_empty());
    }

    #[test]
    fn out_of_range_observations_fail_the_episode() {
        let mut agent = agent(2, 2, PARAMS);
        let mut env = ScriptedEnv {
            resets: VecDeque::from([5]),
            steps: VecDeque::new(),
        };
        assert!(matches!(
            agent.train_episode(&mut env, Some(0)),
            Err(EpisodeError::Agent(QLearningError::InvalidState(5)))
        ));
    }

    #[test]
    fn greedy_policy_shares_probability_between_tied_actions() {
        let mut agent = agent(2, 3, PARAMS);
        agent.update(1, 2, 1.0, 0, true).unwrap();
        let policy = agent.greedy_policy();
        let third = 1.0 / 3.0;
        assert_eq!(policy[0], vec![(0, third), (1, third), (2, third)]);
        assert_eq!(policy[1], vec![(2, 1.0)]);
    }
}
