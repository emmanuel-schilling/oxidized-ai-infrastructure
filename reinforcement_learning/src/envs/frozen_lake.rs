// reinforcement_learning/src/envs/frozen_lake.rs

use crate::core::environment::Environment;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fmt;

/// Action indices, matching Gymnasium's FrozenLake.
pub const LEFT: usize = 0;
pub const DOWN: usize = 1;
pub const RIGHT: usize = 2;
pub const UP: usize = 3;

const NUM_ACTIONS: usize = 4;

/// Gymnasium's default 4x4 map: `S` start, `F` frozen, `H` hole, `G` goal.
pub const MAP_4X4: [&str; 4] = ["SFFF", "FHFH", "FFFH", "HFFG"];

/// The step limit Gymnasium registers for `FrozenLake-v1`.
pub const DEFAULT_MAX_EPISODE_STEPS: usize = 100;

/// One possible outcome of taking an action in a state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition {
    pub probability: f64,
    pub next_state: usize,
    pub reward: f64,
    pub terminated: bool,
}

/// Error type for invalid FrozenLake maps and calls
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrozenLakeError {
    InvalidMap(String),
    InvalidState(usize),
    InvalidAction(usize),
    InvalidPolicy(String),
    ResetNeeded,
}

impl fmt::Display for FrozenLakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMap(msg) => write!(f, "Invalid FrozenLake map: {msg}"),
            Self::InvalidState(state) => write!(f, "Invalid FrozenLake state: {state}"),
            Self::InvalidAction(action) => write!(f, "Invalid FrozenLake action: {action}"),
            Self::InvalidPolicy(msg) => write!(f, "Invalid FrozenLake policy: {msg}"),
            Self::ResetNeeded => write!(f, "FrozenLake step() called before reset()"),
        }
    }
}

impl std::error::Error for FrozenLakeError {}

/// A native Rust port of Gymnasium's `FrozenLake-v1`, including the step limit
/// that `gymnasium.make` adds through its `TimeLimit` wrapper.
///
/// Like Gymnasium, it builds the full transition table up front and samples
/// from it on each step, so the two can be compared entry by entry.
pub struct FrozenLake {
    transitions: Vec<[Vec<Transition>; NUM_ACTIONS]>,
    start_states: Vec<usize>,
    max_episode_steps: Option<usize>,
    state: Option<usize>,
    elapsed_steps: usize,
    rng: Option<ChaCha8Rng>,
}

impl FrozenLake {
    /// Creates an environment from map rows of `S`, `F`, `H` and `G` tiles.
    ///
    /// When `is_slippery` is set, the intended move happens with probability 1/3
    /// and each perpendicular move with probability 1/3. `max_episode_steps`
    /// truncates episodes after that many steps; `None` never truncates.
    pub fn new(
        map: &[&str],
        is_slippery: bool,
        max_episode_steps: Option<usize>,
    ) -> Result<Self, FrozenLakeError> {
        let grid: Vec<&[u8]> = map.iter().map(|row| row.as_bytes()).collect();
        let nrow = grid.len();
        let ncol = grid.first().map_or(0, |row| row.len());
        if nrow == 0 || ncol == 0 {
            return Err(FrozenLakeError::InvalidMap("map is empty".into()));
        }
        if grid.iter().any(|row| row.len() != ncol) {
            return Err(FrozenLakeError::InvalidMap("rows differ in length".into()));
        }
        if let Some(tile) = grid
            .iter()
            .flat_map(|row| row.iter())
            .find(|tile| !b"SFHG".contains(tile))
        {
            return Err(FrozenLakeError::InvalidMap(format!(
                "unknown tile {:?}",
                char::from(*tile)
            )));
        }

        let to_state = |row: usize, col: usize| row * ncol + col;
        let start_states: Vec<usize> = (0..nrow)
            .flat_map(|row| (0..ncol).map(move |col| (row, col)))
            .filter(|&(row, col)| grid[row][col] == b'S')
            .map(|(row, col)| to_state(row, col))
            .collect();
        if start_states.is_empty() {
            return Err(FrozenLakeError::InvalidMap("no start tile".into()));
        }

        // Mirrors Gymnasium's `update_probability_matrix`: moving into a wall
        // leaves the agent in place.
        let outcome = |row: usize, col: usize, action: usize, probability: f64| {
            let (row, col) = match action {
                LEFT => (row, col.saturating_sub(1)),
                DOWN => ((row + 1).min(nrow - 1), col),
                RIGHT => (row, (col + 1).min(ncol - 1)),
                _ => (row.saturating_sub(1), col),
            };
            let tile = grid[row][col];
            Transition {
                probability,
                next_state: to_state(row, col),
                reward: if tile == b'G' { 1.0 } else { 0.0 },
                terminated: tile == b'G' || tile == b'H',
            }
        };

        let mut transitions = Vec::with_capacity(nrow * ncol);
        for (row, tiles) in grid.iter().enumerate() {
            for (col, &tile) in tiles.iter().enumerate() {
                let state = to_state(row, col);
                transitions.push(std::array::from_fn(|action| {
                    if tile == b'G' || tile == b'H' {
                        vec![Transition {
                            probability: 1.0,
                            next_state: state,
                            reward: 0.0,
                            terminated: true,
                        }]
                    } else if is_slippery {
                        // Same order as Gymnasium: (a - 1) % 4, a, (a + 1) % 4
                        [
                            (action + NUM_ACTIONS - 1) % NUM_ACTIONS,
                            action,
                            (action + 1) % NUM_ACTIONS,
                        ]
                        .into_iter()
                        .map(|direction| outcome(row, col, direction, 1.0 / 3.0))
                        .collect()
                    } else {
                        vec![outcome(row, col, action, 1.0)]
                    }
                }));
            }
        }

        Ok(Self {
            transitions,
            start_states,
            max_episode_steps,
            state: None,
            elapsed_steps: 0,
            rng: None,
        })
    }

    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }

    pub fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    /// The possible outcomes of taking `action` in `state`, in Gymnasium's order.
    pub fn transitions(
        &self,
        state: usize,
        action: usize,
    ) -> Result<&[Transition], FrozenLakeError> {
        if action >= NUM_ACTIONS {
            return Err(FrozenLakeError::InvalidAction(action));
        }
        self.transitions
            .get(state)
            .map(|by_action| by_action[action].as_slice())
            .ok_or(FrozenLakeError::InvalidState(state))
    }

    /// The exact expected reward within `max_steps` from the start, when each
    /// state's action is drawn from `policy[state]`, a list of (action,
    /// probability) pairs. The only reward is 1 for reaching the goal, so this
    /// is the probability of reaching it in time.
    pub fn success_probability(
        &self,
        policy: &[Vec<(usize, f64)>],
        max_steps: usize,
    ) -> Result<f64, FrozenLakeError> {
        if policy.len() != self.num_states() {
            return Err(FrozenLakeError::InvalidPolicy(format!(
                "{} states, expected {}",
                policy.len(),
                self.num_states()
            )));
        }
        // reach[s]: expected reward from s within k steps.
        let mut reach = vec![0.0; self.num_states()];
        for _ in 0..max_steps {
            let mut next = vec![0.0; reach.len()];
            for (state, value) in next.iter_mut().enumerate() {
                for &(action, probability) in &policy[state] {
                    for t in self.transitions(state, action)? {
                        let future = if t.terminated {
                            0.0
                        } else {
                            reach[t.next_state]
                        };
                        *value += probability * t.probability * (t.reward + future);
                    }
                }
            }
            reach = next;
        }
        let total: f64 = self.start_states.iter().map(|&start| reach[start]).sum();
        Ok(total / self.start_states.len() as f64)
    }
}

impl Environment for FrozenLake {
    type Observation = usize;
    type Action = usize;
    type Error = FrozenLakeError;

    /// Seeds the RNG when `seed` is given. Otherwise keeps the current RNG,
    /// seeding from OS entropy on first use, as Gymnasium does.
    fn reset(&mut self, seed: Option<u64>) -> Result<Self::Observation, Self::Error> {
        let rng = match seed {
            Some(s) => self.rng.insert(ChaCha8Rng::seed_from_u64(s)),
            None => self.rng.get_or_insert_with(ChaCha8Rng::from_entropy),
        };
        let start = self.start_states[rng.gen_range(0..self.start_states.len())];
        self.state = Some(start);
        self.elapsed_steps = 0;
        Ok(start)
    }

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<(Self::Observation, f64, bool, bool), Self::Error> {
        if action >= NUM_ACTIONS {
            return Err(FrozenLakeError::InvalidAction(action));
        }
        let (Some(state), Some(rng)) = (self.state, self.rng.as_mut()) else {
            return Err(FrozenLakeError::ResetNeeded);
        };

        // Gymnasium's `categorical_sample`: the first outcome whose cumulative
        // probability exceeds a uniform draw in [0, 1).
        let outcomes = &self.transitions[state][action];
        let draw: f64 = rng.sample(Standard);
        let mut cumulative = 0.0;
        let index = outcomes
            .iter()
            .position(|t| {
                cumulative += t.probability;
                cumulative > draw
            })
            .unwrap_or(0);
        let outcome = outcomes[index];

        self.state = Some(outcome.next_state);
        self.elapsed_steps += 1;
        let truncated = self
            .max_episode_steps
            .is_some_and(|max| self.elapsed_steps >= max);

        Ok((
            outcome.next_state,
            outcome.reward,
            outcome.terminated,
            truncated,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lake(is_slippery: bool) -> FrozenLake {
        FrozenLake::new(&MAP_4X4, is_slippery, Some(DEFAULT_MAX_EPISODE_STEPS)).unwrap()
    }

    fn only(next_state: usize, reward: f64, terminated: bool) -> Vec<Transition> {
        vec![Transition {
            probability: 1.0,
            next_state,
            reward,
            terminated,
        }]
    }

    #[test]
    fn default_map_has_16_states_and_starts_at_0() {
        let mut env = lake(true);
        assert_eq!(env.num_states(), 16);
        assert_eq!(env.num_actions(), 4);
        assert_eq!(env.reset(Some(0)), Ok(0));
    }

    #[test]
    fn invalid_maps_are_rejected() {
        for map in [
            &[][..],
            &[""][..],
            &["SF", "F"][..],
            &["SX"][..],
            &["FG"][..],
        ] {
            assert!(
                matches!(
                    FrozenLake::new(map, true, None),
                    Err(FrozenLakeError::InvalidMap(_))
                ),
                "map {map:?} should be rejected"
            );
        }
    }

    #[test]
    fn non_slippery_moves_clamp_at_walls() {
        let env = lake(false);
        assert_eq!(env.transitions(0, LEFT).unwrap(), only(0, 0.0, false));
        assert_eq!(env.transitions(0, UP).unwrap(), only(0, 0.0, false));
        assert_eq!(env.transitions(0, RIGHT).unwrap(), only(1, 0.0, false));
        assert_eq!(env.transitions(0, DOWN).unwrap(), only(4, 0.0, false));
    }

    #[test]
    fn holes_end_with_no_reward_and_the_goal_with_reward_1() {
        let env = lake(false);
        assert_eq!(env.transitions(1, DOWN).unwrap(), only(5, 0.0, true));
        assert_eq!(env.transitions(14, RIGHT).unwrap(), only(15, 1.0, true));
    }

    #[test]
    fn slippery_outcomes_follow_gymnasium_order() {
        let env = lake(true);
        // From state 6, intending DOWN: LEFT into hole 5, DOWN to 10, RIGHT into hole 7.
        let third = 1.0 / 3.0;
        let expected = vec![
            Transition {
                probability: third,
                next_state: 5,
                reward: 0.0,
                terminated: true,
            },
            Transition {
                probability: third,
                next_state: 10,
                reward: 0.0,
                terminated: false,
            },
            Transition {
                probability: third,
                next_state: 7,
                reward: 0.0,
                terminated: true,
            },
        ];
        assert_eq!(env.transitions(6, DOWN).unwrap(), expected);
    }

    #[test]
    fn hole_and_goal_states_loop_to_themselves() {
        for is_slippery in [false, true] {
            let env = lake(is_slippery);
            for action in 0..NUM_ACTIONS {
                assert_eq!(env.transitions(5, action).unwrap(), only(5, 0.0, true));
                assert_eq!(env.transitions(15, action).unwrap(), only(15, 0.0, true));
            }
        }
    }

    #[test]
    fn transitions_rejects_out_of_range_arguments() {
        let env = lake(true);
        assert_eq!(
            env.transitions(16, LEFT),
            Err(FrozenLakeError::InvalidState(16))
        );
        assert_eq!(
            env.transitions(0, 4),
            Err(FrozenLakeError::InvalidAction(4))
        );
    }

    #[test]
    fn non_slippery_episode_reaches_the_goal() {
        let mut env = lake(false);
        env.reset(Some(0)).unwrap();
        let path = [(DOWN, 4), (DOWN, 8), (RIGHT, 9), (DOWN, 13), (RIGHT, 14)];
        for (action, state) in path {
            assert_eq!(env.step(action), Ok((state, 0.0, false, false)));
        }
        assert_eq!(env.step(RIGHT), Ok((15, 1.0, true, false)));
    }

    #[test]
    fn episodes_truncate_at_the_step_limit_and_reset_clears_the_count() {
        let mut env = lake(false);
        env.reset(Some(0)).unwrap();
        for _ in 1..DEFAULT_MAX_EPISODE_STEPS {
            assert_eq!(env.step(LEFT), Ok((0, 0.0, false, false)));
        }
        assert_eq!(env.step(LEFT), Ok((0, 0.0, false, true)));

        env.reset(None).unwrap();
        assert_eq!(env.step(LEFT), Ok((0, 0.0, false, false)));
    }

    #[test]
    fn without_a_step_limit_episodes_never_truncate() {
        let mut env = FrozenLake::new(&MAP_4X4, false, None).unwrap();
        env.reset(Some(0)).unwrap();
        for _ in 0..1000 {
            assert_eq!(env.step(LEFT), Ok((0, 0.0, false, false)));
        }
    }

    #[test]
    fn a_step_can_be_both_terminated_and_truncated() {
        let mut env = FrozenLake::new(&["SG"], false, Some(1)).unwrap();
        env.reset(Some(0)).unwrap();
        assert_eq!(env.step(RIGHT), Ok((1, 1.0, true, true)));
    }

    #[test]
    fn step_rejects_invalid_calls() {
        let mut env = lake(true);
        assert_eq!(env.step(LEFT), Err(FrozenLakeError::ResetNeeded));
        env.reset(Some(0)).unwrap();
        assert_eq!(env.step(4), Err(FrozenLakeError::InvalidAction(4)));
    }

    /// Runs a fixed action sequence, resetting without a seed after each episode.
    fn trajectory(env: &mut FrozenLake, seed: u64) -> Vec<(usize, f64, bool, bool)> {
        env.reset(Some(seed)).unwrap();
        (0..500)
            .map(|i| {
                let result = env.step(i % NUM_ACTIONS).unwrap();
                if result.2 || result.3 {
                    env.reset(None).unwrap();
                }
                result
            })
            .collect()
    }

    #[test]
    fn the_same_seed_gives_the_same_trajectory() {
        assert_eq!(
            trajectory(&mut lake(true), 7),
            trajectory(&mut lake(true), 7)
        );
        assert_ne!(
            trajectory(&mut lake(true), 7),
            trajectory(&mut lake(true), 8)
        );
    }

    #[test]
    fn reset_without_a_seed_works_before_any_seeded_reset() {
        let mut env = lake(true);
        assert_eq!(env.reset(None), Ok(0));
        assert!(env.step(DOWN).is_ok());
    }

    #[test]
    fn slippery_outcomes_are_equally_likely() {
        // From state 0, intending DOWN: LEFT stays at 0, DOWN goes to 4, RIGHT to 1.
        const DRAWS: usize = 30_000;
        let mut env = lake(true);
        env.reset(Some(11)).unwrap();
        let mut counts = [0usize; 16];
        for _ in 0..DRAWS {
            counts[env.step(DOWN).unwrap().0] += 1;
            env.reset(None).unwrap();
        }
        // About 5 standard deviations of a 1/3 proportion over 30,000 draws.
        for state in [0, 1, 4] {
            let frequency = counts[state] as f64 / DRAWS as f64;
            assert!(
                (frequency - 1.0 / 3.0).abs() < 0.014,
                "state {state}: {frequency}"
            );
        }
        assert_eq!(counts[0] + counts[1] + counts[4], DRAWS);
    }

    fn uniform_policy(actions: &[usize]) -> Vec<(usize, f64)> {
        let probability = 1.0 / actions.len() as f64;
        actions.iter().map(|&a| (a, probability)).collect()
    }

    #[test]
    fn success_probability_follows_the_policy() {
        // Down, down, right, down, right, right: a shortest path to the goal.
        let mut path = vec![uniform_policy(&[LEFT]); 16];
        for (state, action) in [
            (0, DOWN),
            (4, DOWN),
            (8, RIGHT),
            (9, DOWN),
            (13, RIGHT),
            (14, RIGHT),
        ] {
            path[state] = uniform_policy(&[action]);
        }
        let lake = lake(false);
        assert_eq!(lake.success_probability(&path, 100), Ok(1.0));
        // Six steps are needed, so five are not enough.
        assert_eq!(lake.success_probability(&path, 5), Ok(0.0));

        // Down from the start, then right into the hole at state 5.
        path[4] = uniform_policy(&[RIGHT]);
        assert_eq!(lake.success_probability(&path, 100), Ok(0.0));
    }

    #[test]
    fn a_uniformly_random_policy_rarely_crosses_the_slippery_lake() {
        let random = vec![uniform_policy(&[LEFT, DOWN, RIGHT, UP]); 16];
        let probability = lake(true).success_probability(&random, 100).unwrap();
        assert!(0.005 < probability && probability < 0.05, "{probability}");
    }

    #[test]
    fn success_probability_rejects_invalid_policies() {
        let lake = lake(false);
        assert!(matches!(
            lake.success_probability(&vec![uniform_policy(&[LEFT]); 15], 100),
            Err(FrozenLakeError::InvalidPolicy(_))
        ));
        let mut policy = vec![uniform_policy(&[LEFT]); 16];
        policy[3] = uniform_policy(&[4]);
        assert_eq!(
            lake.success_probability(&policy, 100),
            Err(FrozenLakeError::InvalidAction(4))
        );
    }
}
