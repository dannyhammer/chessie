/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![doc = include_str!("../README.md")]

/// All things related to Bitboards.
pub mod bitboard;
/// Enums for piece kinds, colors, and a struct for a chess piece.
pub mod piece;
/// Squares on a chessboard (including files and ranks).
pub mod square;
/// Misc utility functions and constants, as well as magic bitboard generation.
pub mod utils;

pub use bitboard::*;
pub use piece::*;
pub use square::*;
pub use utils::*;

/// Re-exports all the things you'll need.
pub mod prelude {
    pub use crate::bitboard::*;
    pub use crate::piece::*;
    pub use crate::square::*;
    pub use crate::utils::*;
}
