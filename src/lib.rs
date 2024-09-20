pub use types::*;

/// High-level abstraction of the game of chess, including movable pieces, legality checks, game state, etc.
///
/// You probably want to look here.
pub mod game;
/// All code related to generating moves (legal and pseudo-legal) for pieces on a board.
pub mod movegen;
/// Enums and structs for modeling the movement of a piece on a chessboard.
pub mod moves;
/// Utility function for performance testing.
pub mod perft;
/// A chessboard, complete with piece placements, turn counters, and game state information.
pub mod position;
/// Pseudo-random number generation, written to be usable in `const` settings.
///
/// Primarily for Zobrist hashing.
pub mod prng;
/// Zobrist keys for hashing chess positions.
pub mod zobrist;

pub use game::*;
pub use movegen::*;
pub use moves::*;
pub use perft::*;
pub use position::*;
pub use prng::*;
pub use zobrist::*;

/// Re-exports all the things you'll need.
pub mod prelude {
    pub use crate::game::*;
    pub use crate::movegen::*;
    pub use crate::moves::*;
    pub use crate::perft::*;
    pub use crate::position::*;
    pub use crate::prng::*;
    pub use crate::utils::*;
    pub use crate::zobrist::*;
}
