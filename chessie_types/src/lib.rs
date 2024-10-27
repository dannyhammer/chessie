/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![doc = include_str!("../README.md")]

/// All things related to Bitboards.
mod bitboard;
/// Enums for piece kinds, colors, and a struct for a chess piece.
mod piece;
/// Pseudo-random number generation, written to be usable in `const` settings.
///
/// Primarily for Zobrist hashing and magic generation.
mod prng;
/// Squares on a chessboard (including files and ranks).
mod square;
/// Misc utility functions and constants, as well as magic bitboard generation.
mod utils;

pub use bitboard::*;
pub use piece::*;
pub use prng::*;
pub use square::*;
pub use utils::*;

/// Re-exports all the things you'll need.
pub mod prelude {
    pub use crate::bitboard::*;
    pub use crate::piece::*;
    pub use crate::prng::*;
    pub use crate::square::*;
    pub use crate::utils::*;
}
