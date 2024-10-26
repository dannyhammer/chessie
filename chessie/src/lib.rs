/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![doc = include_str!("../README.md")]

pub use chessie_types::*;

/// High-level abstraction of the game of chess, including movable pieces, legality checks, game state, etc.
mod game;
/// All code related to generating moves (legal and pseudo-legal) for pieces on a board.
mod movegen;
/// Enums and structs for modeling the movement of a piece on a chessboard.
mod moves;
/// Utility function for performance testing.
mod perft;
/// A chessboard, complete with piece placements, turn counters, and game state information.
mod position;
/// Zobrist keys for hashing chess positions.
mod zobrist;

pub use game::*;
pub use movegen::*;
pub use moves::*;
pub use perft::*;
pub use position::*;
pub use zobrist::*;

/// Re-exports all the things you'll need.
pub mod prelude {
    pub use crate::game::*;
    pub use crate::movegen::*;
    pub use crate::moves::*;
    pub use crate::perft::*;
    pub use crate::position::*;
    pub use crate::zobrist::*;
    pub use chessie_types::*;
}
