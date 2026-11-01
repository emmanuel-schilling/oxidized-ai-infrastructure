// reinforcement_learning/src/core/environment.rs

use std::error::Error;

/// The foundational contract for all simulation environments.
///
/// Using associated types (`Observation`, `Action`, and `Error`) ensures that solution
/// algorithms remain completely decoupled from the underlying environment implementation
/// (e.g., discrete gridworlds vs. continuous tensor spaces) and can propagate runtime errors safely.
pub trait Environment {
    /// The type representing the state or observation space (e.g., `usize`, `ndarray::Array1<f64>`)
    type Observation;

    /// The type representing the action space (e.g., `usize` for discrete actions, `ndarray::Array1<f64>` for continuous)
    type Action;

    /// The error type returned when environment transitions or resets fail
    type Error: Error + Send + Sync + 'static;

    /// Resets the environment to an initial state and returns the initial observation.
    ///
    /// Accepts an optional `seed` parameter to enable deterministic, reproducible rollouts.
    fn reset(&mut self, seed: Option<u64>) -> Result<Self::Observation, Self::Error>;

    /// Steps the environment forward by one time step using the given action.
    ///
    /// Returns a tuple containing:
    /// 1. `Self::Observation` - The next state observation.
    /// 2. `f64` - The reward scalar obtained from the transition.
    /// 3. `bool` - The `terminated` flag (e.g., reaching a terminal goal or falling over).
    /// 4. `bool` - The `truncated` flag (e.g., hitting a maximum episode step limit).
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<(Self::Observation, f64, bool, bool), Self::Error>;
}
