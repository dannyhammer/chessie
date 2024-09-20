/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::{fmt, str::FromStr};

use anyhow::{anyhow, Result};

use super::{Piece, PieceKind, Position, Square};

/// Represents the different kinds of moves that can be made during a chess game.
///
/// Internally, these are represented by bit flags, which allows a compact representation of the [`Move`] struct.
/// You do not need to know the bit flag values. They are only relevant internally.
/// If you care, though, they are fetched from the [chess programming wiki](https://www.chessprogramming.org/Encoding_Moves#From-To_Based).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum MoveKind {
    /// Involves only a single piece moving from one location to another, and does not change the quantity or kind of any pieces on the board.
    Quiet = 0 << Move::FLG_BITS,

    /// A special case on a Pawn's first move, wherein it can advance two squares forward.
    PawnDoublePush = 1 << Move::FLG_BITS,

    /// Involves the King and a Rook sliding past each other on the King's side of the board.
    ShortCastle = 2 << Move::FLG_BITS,

    /// Involves the King and a Rook sliding past each other on the Queen's side of the board.
    LongCastle = 3 << Move::FLG_BITS,

    /// Involves a piece moving onto a square occupied by an opponent's piece, removing it from the board.
    Capture = 4 << Move::FLG_BITS,

    /// A special variant of capturing that occurs when a Pawn executes En Passant.
    EnPassantCapture = 5 << Move::FLG_BITS,

    /// Involves a Pawn reaching the opponent's side of the board (rank 8 for White, rank 1 for Black) and becoming a [`PieceKind::Knight`].
    PromoteKnight = 8 << Move::FLG_BITS,

    /// Involves a Pawn reaching the opponent's side of the board (rank 8 for White, rank 1 for Black) and becoming a [`PieceKind::Bishop`].
    PromoteBishop = 9 << Move::FLG_BITS,

    /// Involves a Pawn reaching the opponent's side of the board (rank 8 for White, rank 1 for Black) and becoming a [`PieceKind::Rook`].
    PromoteRook = 10 << Move::FLG_BITS,

    /// Involves a Pawn reaching the opponent's side of the board (rank 8 for White, rank 1 for Black) and becoming a [`PieceKind::Queen`].
    PromoteQueen = 11 << Move::FLG_BITS,

    /// Involves a Pawn moving onto a square on the opponent's side of the board that is occupied by an opponent's piece, removing it from the board, and promoting this Pawn to a [`PieceKind::Knight`].
    CaptureAndPromoteKnight = 12 << Move::FLG_BITS,

    /// Involves a Pawn moving onto a square on the opponent's side of the board that is occupied by an opponent's piece, removing it from the board, and promoting this Pawn to a [`PieceKind::Bishop`].
    CaptureAndPromoteBishop = 13 << Move::FLG_BITS,

    /// Involves a Pawn moving onto a square on the opponent's side of the board that is occupied by an opponent's piece, removing it from the board, and promoting this Pawn to a [`PieceKind::Rook`].
    CaptureAndPromoteRook = 14 << Move::FLG_BITS,

    /// Involves a Pawn moving onto a square on the opponent's side of the board that is occupied by an opponent's piece, removing it from the board, and promoting this Pawn to a [`PieceKind::Queen`].
    CaptureAndPromoteQueen = 15 << Move::FLG_BITS,
}

impl MoveKind {
    /// Creates a new [`MoveKind`] that is a promotion to the provided [`PieceKind`].
    ///
    /// # Panics
    /// This function will panic if `promotion` is not a Knight, Bishop, Rook, or Queen.
    pub fn promotion(promotion: PieceKind) -> Self {
        match promotion {
            PieceKind::Knight => Self::PromoteKnight,
            PieceKind::Bishop => Self::PromoteBishop,
            PieceKind::Rook => Self::PromoteRook,
            PieceKind::Queen => Self::PromoteQueen,
            _ => unreachable!(),
        }
    }

    /// Creates a new [`MoveKind`] that is a capture and promotion to the provided [`PieceKind`].
    ///
    /// # Panics
    /// This function will panic if `promotion` is not a Knight, Bishop, Rook, or Queen.
    pub fn promotion_capture(promotion: PieceKind) -> Self {
        match promotion {
            PieceKind::Knight => Self::CaptureAndPromoteKnight,
            PieceKind::Bishop => Self::CaptureAndPromoteBishop,
            PieceKind::Rook => Self::CaptureAndPromoteRook,
            PieceKind::Queen => Self::CaptureAndPromoteQueen,
            _ => unreachable!(),
        }
    }

    /// Determines the appropriate [`MoveKind`] for moving the `piece` at `from` onto `to`, within the provided `position`.
    ///
    /// If `promotion` was provided and other parameters specify that this is a pawn moving to the eighth rank,
    /// this will yield a promotion variant that promotes to the [`PieceKind`] specified by `promotion`.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Position, MoveKind, Piece, PieceKind, Square};
    /// let pos = Position::default();
    /// let kind = MoveKind::new(Piece::WHITE_PAWN, Square::E2, Square::E4, &pos, None);
    /// assert_eq!(kind, MoveKind::PawnDoublePush);
    /// ```
    pub fn new(
        piece: Piece,
        from: Square,
        to: Square,
        position: &Position,
        promotion: Option<PieceKind>,
    ) -> Self {
        // Extract information about the piece being moved
        let color = piece.color();

        // By default, it's either a quiet or a capture.
        let mut kind = if position.has(to) {
            Self::Capture
        } else {
            Self::Quiet
        };

        // The MoveKind depends on what kind of piece is being moved and where
        if piece.is_pawn() {
            // If a promotion was provided, it's a promotion of some kind
            if let Some(promotion) = promotion {
                // If this move also captures, it's a capture-promote
                if kind == Self::Capture {
                    kind = Self::promotion_capture(promotion);
                } else {
                    kind = Self::promotion(promotion);
                }
            }
            // If this pawn is moving to the en passant square, it's en passant
            else if Some(to) == position.ep_square() {
                kind = Self::EnPassantCapture;
            }
            // If the Pawn is moving two ranks, it's a double push
            else if from.rank().abs_diff(to.rank()) == 2 {
                kind = Self::PawnDoublePush;
            }
        } else if piece.is_king() && from == Square::E1.rank_relative_to(color) {
            if to == Square::G1.rank_relative_to(color) {
                kind = Self::ShortCastle;
            } else if to == Square::C1.rank_relative_to(color) {
                kind = Self::LongCastle;
            }
        }

        kind
    }

    /// Fetches the [`PieceKind`] to promote to, if `self` is a promotion.
    pub fn get_promotion(&self) -> Option<PieceKind> {
        match self {
            Self::PromoteQueen | Self::CaptureAndPromoteQueen => Some(PieceKind::Queen),
            Self::PromoteKnight | Self::CaptureAndPromoteKnight => Some(PieceKind::Knight),
            Self::PromoteRook | Self::CaptureAndPromoteRook => Some(PieceKind::Rook),
            Self::PromoteBishop | Self::CaptureAndPromoteBishop => Some(PieceKind::Bishop),
            _ => None,
        }
    }
}

/// Represents a move made on a chess board, including whether a piece is to be promoted.
///
/// Internally encoded using the following bit pattern:
/// ```text
///     0000 000000 000000
///      |     |      |
///      |     |      +- Source square of the move.
///      |     +- Target square of the move.
///      +- Special flags for promotion, castling, etc.
/// ```
///
/// Flags are fetched directly from the [Chess Programming Wiki](https://www.chessprogramming.org/Encoding_Moves#From-To_Based).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Move(u16);

impl Move {
    /// Mask for the source ("from") bits.
    const SRC_MASK: u16 = 0b0000_0000_0011_1111;
    /// Mask for the destination ("to") bits.
    const DST_MASK: u16 = 0b0000_1111_1100_0000;
    /// Mask for the flag (promotions, captures, etc.) bits.
    const FLG_MASK: u16 = 0b1111_0000_0000_0000;
    /// Start index of destination bits.
    const DST_BITS: u16 = 6;
    /// Start index of flag bits.
    const FLG_BITS: u16 = 12;

    // const FLAG_QUIET: u16 = 0 << Self::FLG_BITS;
    const FLAG_PAWN_DOUBLE: u16 = 1 << Self::FLG_BITS;
    const FLAG_CASTLE_SHORT: u16 = 2 << Self::FLG_BITS;
    const FLAG_CASTLE_LONG: u16 = 3 << Self::FLG_BITS;
    const FLAG_CAPTURE: u16 = 4 << Self::FLG_BITS;
    const FLAG_EP_CAPTURE: u16 = 5 << Self::FLG_BITS;
    const FLAG_PROMO_KNIGHT: u16 = 8 << Self::FLG_BITS;
    const FLAG_PROMO_BISHOP: u16 = 9 << Self::FLG_BITS;
    const FLAG_PROMO_ROOK: u16 = 10 << Self::FLG_BITS;
    const FLAG_PROMO_QUEEN: u16 = 11 << Self::FLG_BITS;
    const FLAG_CAPTURE_PROMO_KNIGHT: u16 = 12 << Self::FLG_BITS;
    const FLAG_CAPTURE_PROMO_BISHOP: u16 = 13 << Self::FLG_BITS;
    const FLAG_CAPTURE_PROMO_ROOK: u16 = 14 << Self::FLG_BITS;
    const FLAG_CAPTURE_PROMO_QUEEN: u16 = 15 << Self::FLG_BITS;

    /// Creates a new [`Move`] from the given [`Square`]s and a [`MoveKind`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind, PieceKind};
    /// let e2e4 = Move::new(Square::E2, Square::E4, MoveKind::PawnDoublePush);
    /// assert_eq!(e2e4.to_string(), "e2e4");
    ///
    /// let e7e8n = Move::new(Square::E7, Square::E8, MoveKind::promotion(PieceKind::Knight));
    /// assert_eq!(e7e8n.to_string(), "e7e8n");
    /// ```
    pub fn new(from: Square, to: Square, kind: MoveKind) -> Self {
        Self(kind as u16 | (to.inner() as u16) << Self::DST_BITS | from.inner() as u16)
    }

    /// Creates a new [`Move`] from the given [`Square`]s that does not promote a piece.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square};
    /// let e2e3 = Move::new_quiet(Square::E2, Square::E3);
    /// assert_eq!(e2e3.to_string(), "e2e3");
    /// ```
    pub fn new_quiet(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveKind::Quiet)
    }

    /// Creates an "illegal" [`Move`], representing moving a piece to and from the same [`Square`].
    ///
    /// # Example
    /// ```
    /// # use chessie::Move;
    /// let illegal = Move::illegal();
    /// assert_eq!(illegal.to_string(), "a1a1");
    /// ```
    pub const fn illegal() -> Self {
        Self(0)
    }

    /// Fetches the source (or "from") part of this [`Move`], as a [`Square`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind};
    /// let e2e4 = Move::new(Square::E2, Square::E4, MoveKind::PawnDoublePush);
    /// let from = e2e4.from();
    /// assert_eq!(from, Square::E2);
    /// ```
    pub const fn from(&self) -> Square {
        Square::from_bits_unchecked((self.0 & Self::SRC_MASK) as u8)
    }

    /// Fetches the destination (or "to") part of this [`Move`], as a [`Square`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind};
    /// let e2e4 = Move::new(Square::E2, Square::E4, MoveKind::PawnDoublePush);
    /// let to = e2e4.to();
    /// assert_eq!(to, Square::E4);
    /// ```
    pub const fn to(&self) -> Square {
        Square::from_bits_unchecked(((self.0 & Self::DST_MASK) >> Self::DST_BITS) as u8)
    }

    /// Fetches the [`MoveKind`] part of this [`Move`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, MoveKind, PieceKind, Square};
    /// let e7e8q = Move::new(Square::E7, Square::E8, MoveKind::promotion(PieceKind::Queen));
    /// assert_eq!(e7e8q.kind(), MoveKind::promotion(PieceKind::Queen));
    /// ```
    pub fn kind(&self) -> MoveKind {
        // Safety: Since a `Move` can ONLY be constructed through the public API,
        // any instance of a `Move` is guaranteed to have a valid bit pattern for its `MoveKind`.
        unsafe { std::mem::transmute(self.0 & Self::FLG_MASK) }
    }

    /// Fetches the parts of this [`Move`] in a tuple of `(from, to, kind)`.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, MoveKind, PieceKind, Square};
    /// let e7e8q = Move::new(Square::E7, Square::E8, MoveKind::promotion(PieceKind::Queen));
    /// let (from, to, kind) = e7e8q.parts();
    /// assert_eq!(from, Square::E7);
    /// assert_eq!(to, Square::E8);
    /// assert_eq!(kind, MoveKind::promotion(PieceKind::Queen));
    /// ```
    pub fn parts(&self) -> (Square, Square, MoveKind) {
        (self.from(), self.to(), self.kind())
    }

    /// Returns `true` if this [`Move`] is a capture of any kind.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind, PieceKind, Position, FEN_KIWIPETE};
    /// let position = Position::from_fen(FEN_KIWIPETE).unwrap();
    /// let e5f7 = Move::from_uci(&position, "e5f7").unwrap();
    /// assert_eq!(e5f7.is_capture(), true);
    /// ```
    pub const fn is_capture(&self) -> bool {
        self.0 & Self::FLAG_CAPTURE != 0
    }

    /// Returns `true` if this [`Move`] is en passant.
    pub const fn is_en_passant(&self) -> bool {
        (self.0 & Self::FLG_MASK) ^ Self::FLAG_EP_CAPTURE == 0
    }

    /// Returns `true` if this [`Move`] is a short (kingside) castle.
    pub const fn is_short_castle(&self) -> bool {
        (self.0 & Self::FLG_MASK) ^ Self::FLAG_CASTLE_SHORT == 0
    }

    /// Returns `true` if this [`Move`] is a long (queenside) castle.
    pub const fn is_long_castle(&self) -> bool {
        (self.0 & Self::FLG_MASK) ^ Self::FLAG_CASTLE_LONG == 0
    }

    /// Returns `true` if this [`Move`] is a short (kingside) or long (queenside) castle.
    pub const fn is_castle(&self) -> bool {
        self.is_short_castle() || self.is_long_castle()
    }

    /// Returns `true` if this [`Move`] is a long (queenside) castle.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind};
    /// let e2e4 = Move::new(Square::E2, Square::E4, MoveKind::PawnDoublePush);
    /// assert_eq!(e2e4.is_pawn_double_push(), true);
    /// ```
    pub const fn is_pawn_double_push(&self) -> bool {
        (self.0 & Self::FLG_MASK) ^ Self::FLAG_PAWN_DOUBLE == 0
    }

    /// Returns `true` if this [`Move`] is a promotion of any kind.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, MoveKind, PieceKind, Square};
    /// let e7e8q = Move::new(Square::E7, Square::E8, MoveKind::promotion(PieceKind::Queen));
    /// assert_eq!(e7e8q.is_promotion(), true);
    ///
    /// let e7e8q = Move::new(Square::E7, Square::E8, MoveKind::promotion_capture(PieceKind::Queen));
    /// assert_eq!(e7e8q.is_promotion(), true);
    /// ```
    pub const fn is_promotion(&self) -> bool {
        // The flag bit for "promotion" is the most-significant bit.
        // Internally, FLAG_PROMO_KNIGHT has flag bits `1000`, so we can use it as a mask for promotions.
        self.0 & Self::FLAG_PROMO_KNIGHT != 0
    }

    /// Returns `true` if this [`Move`] is a capture of any kind.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind, PieceKind, Position};
    /// // An sample test position for discovering promotion bugs.
    /// let position = Position::from_fen("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ").unwrap();
    /// let b7c8b = Move::from_uci(&position, "b7c8b").unwrap();
    /// assert_eq!(b7c8b.promotion(), Some(PieceKind::Bishop));
    /// ```
    pub fn promotion(&self) -> Option<PieceKind> {
        match self.0 & Self::FLG_MASK {
            Self::FLAG_PROMO_QUEEN | Self::FLAG_CAPTURE_PROMO_QUEEN => Some(PieceKind::Queen),
            Self::FLAG_PROMO_KNIGHT | Self::FLAG_CAPTURE_PROMO_KNIGHT => Some(PieceKind::Knight),
            Self::FLAG_PROMO_ROOK | Self::FLAG_CAPTURE_PROMO_ROOK => Some(PieceKind::Rook),
            Self::FLAG_PROMO_BISHOP | Self::FLAG_CAPTURE_PROMO_BISHOP => Some(PieceKind::Bishop),
            _ => None,
        }
    }

    /// Returns `true` if this move is formatted properly according to [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use chessie::Move;
    /// assert_eq!(Move::is_uci("b7c8b"), true);
    /// assert_eq!(Move::is_uci("a1a1"), true);
    /// assert_eq!(Move::is_uci("xj9"), false);
    /// ```
    pub fn is_uci(input: &str) -> bool {
        let Some(from) = input.get(0..2) else {
            return false;
        };
        let Some(to) = input.get(2..4) else {
            return false;
        };

        let is_ok = Square::from_uci(from).is_ok() && Square::from_uci(to).is_ok();

        if let Some(promote) = input.get(4..5) {
            is_ok && PieceKind::from_str(promote).is_ok()
        } else {
            is_ok
        }
    }
    /// Creates a [`Move`] from a string, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation, extracting extra info from the provided [`Position`]
    ///
    /// Will return a [`anyhow::Error`] if the string is invalid in any way.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind, PieceKind, Position};
    /// // A sample test position for discovering promotion bugs.
    /// let position = Position::from_fen("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ").unwrap();
    /// let b7c8b = Move::from_uci(&position, "b7c8b");
    /// assert!(b7c8b.is_ok());
    /// assert_eq!(b7c8b.unwrap(), Move::new(Square::B7, Square::C8, MoveKind::promotion_capture(PieceKind::Bishop)));
    /// ```
    pub fn from_uci(position: &Position, uci: &str) -> Result<Self> {
        // Extract the to/from squares
        let from = uci.get(0..2).ok_or(anyhow!(
            "Move str must contain a `from` square. Got {uci:?}"
        ))?;
        let to = uci
            .get(2..4)
            .ok_or(anyhow!("Move str must contain a `to` square. Got {uci:?}"))?;

        let from = Square::from_uci(from)?;
        let to = Square::from_uci(to)?;

        // Extract information about the piece being moved
        let piece = position.board().piece_at(from).ok_or(anyhow!(
            "No piece found at {from} when parsing {uci:?} on position {position}"
        ))?;

        // If there is a promotion char, attempt to convert it to a PieceKind
        let promotion = uci.get(4..5).map(PieceKind::from_str).transpose()?;

        // The MoveKind depends on what kind of piece is being moved and where
        let kind = MoveKind::new(piece, from, to, position, promotion);

        Ok(Self::new(from, to, kind))
    }

    /// Converts this [`Move`] to a string, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// Please note that promotions are capitalized by default.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Move, Square, MoveKind, PieceKind};
    /// let e7e8Q = Move::new(Square::E7, Square::E8, MoveKind::promotion(PieceKind::Queen));
    /// assert_eq!(e7e8Q.to_uci(), "e7e8q");
    /// ```
    pub fn to_uci(&self) -> String {
        if let Some(promote) = self.promotion() {
            format!("{}{}{}", self.from(), self.to(), promote)
        } else {
            format!("{}{}", self.from(), self.to())
        }
    }
}

impl fmt::Display for Move {
    /// A [`Move`] is displayed in its UCI format.
    ///
    /// See [`Move::to_uci`] for more.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:?})", self.to_uci(), self.kind())
    }
}

impl Default for Move {
    /// A "default" move is an illegal move. See [`Move::illegal`]
    ///
    /// This is mostly just to satisfy the compiler, and should never be used in a real scenario.
    fn default() -> Self {
        Self::illegal()
    }
}

impl<T: AsRef<str>> PartialEq<T> for Move {
    fn eq(&self, other: &T) -> bool {
        self.to_uci().eq(other.as_ref())
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_move_is_capture() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_capture());
        assert!(!Move::new(from, to, MoveKind::ShortCastle).is_capture());
        assert!(!Move::new(from, to, MoveKind::LongCastle).is_capture());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_capture());
        assert!(Move::new(from, to, MoveKind::Capture).is_capture());
        assert!(Move::new(from, to, MoveKind::EnPassantCapture).is_capture());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_capture());
        assert!(Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_capture());
    }

    #[test]
    fn test_move_is_en_passant() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_en_passant());
        assert!(!Move::new(from, to, MoveKind::ShortCastle).is_en_passant());
        assert!(!Move::new(from, to, MoveKind::LongCastle).is_en_passant());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_en_passant());
        assert!(!Move::new(from, to, MoveKind::Capture).is_en_passant());
        assert!(Move::new(from, to, MoveKind::EnPassantCapture).is_en_passant());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_en_passant());
        assert!(
            !Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_en_passant()
        );
    }

    #[test]
    fn test_move_is_short_castle() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_short_castle());
        assert!(Move::new(from, to, MoveKind::ShortCastle).is_short_castle());
        assert!(!Move::new(from, to, MoveKind::LongCastle).is_short_castle());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_short_castle());
        assert!(!Move::new(from, to, MoveKind::Capture).is_short_castle());
        assert!(!Move::new(from, to, MoveKind::EnPassantCapture).is_short_castle());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_short_castle());
        assert!(
            !Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_short_castle()
        );
    }

    #[test]
    fn test_move_is_long_castle() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_long_castle());
        assert!(!Move::new(from, to, MoveKind::ShortCastle).is_long_castle());
        assert!(Move::new(from, to, MoveKind::LongCastle).is_long_castle());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_long_castle());
        assert!(!Move::new(from, to, MoveKind::Capture).is_long_castle());
        assert!(!Move::new(from, to, MoveKind::EnPassantCapture).is_long_castle());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_long_castle());
        assert!(
            !Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_long_castle()
        );
    }

    #[test]
    fn test_move_is_castle() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_castle());
        assert!(Move::new(from, to, MoveKind::ShortCastle).is_castle());
        assert!(Move::new(from, to, MoveKind::LongCastle).is_castle());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_castle());
        assert!(!Move::new(from, to, MoveKind::Capture).is_castle());
        assert!(!Move::new(from, to, MoveKind::EnPassantCapture).is_castle());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_castle());
        assert!(!Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_castle());
    }

    #[test]
    fn test_move_is_pawn_double_push() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_pawn_double_push());
        assert!(!Move::new(from, to, MoveKind::ShortCastle).is_pawn_double_push());
        assert!(!Move::new(from, to, MoveKind::LongCastle).is_pawn_double_push());
        assert!(Move::new(from, to, MoveKind::PawnDoublePush).is_pawn_double_push());
        assert!(!Move::new(from, to, MoveKind::Capture).is_pawn_double_push());
        assert!(!Move::new(from, to, MoveKind::EnPassantCapture).is_pawn_double_push());
        assert!(!Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_pawn_double_push());
        assert!(
            !Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen))
                .is_pawn_double_push()
        );
    }

    #[test]
    fn test_move_is_promotion() {
        let (from, to) = (Square::A1, Square::H8);
        assert!(!Move::new(from, to, MoveKind::Quiet).is_promotion());
        assert!(!Move::new(from, to, MoveKind::ShortCastle).is_promotion());
        assert!(!Move::new(from, to, MoveKind::LongCastle).is_promotion());
        assert!(!Move::new(from, to, MoveKind::PawnDoublePush).is_promotion());
        assert!(!Move::new(from, to, MoveKind::Capture).is_promotion());
        assert!(!Move::new(from, to, MoveKind::EnPassantCapture).is_promotion());
        assert!(Move::new(from, to, MoveKind::promotion(PieceKind::Queen)).is_promotion());
        assert!(Move::new(from, to, MoveKind::promotion_capture(PieceKind::Queen)).is_promotion());
    }

    /// Helper function to assert that the `uci` move is parsed as `expected` on the position created from `fen`.
    fn test_move_parse(fen: &str, uci: &str, expected: Move) {
        let pos = fen.parse().unwrap();

        let mv = Move::from_uci(&pos, uci);
        assert!(mv.is_ok(), "{}", mv.unwrap_err());
        assert_eq!(mv.unwrap(), expected);
    }

    #[test]
    fn test_move_parsing() {
        // We can test all moves except castling with Pawns
        let pawn_fen = "2n1k3/1P6/8/5pP1/5n2/2P1P3/P7/4K3 w - f6 0 1";

        // Pawn single push
        let mv = Move::new(Square::A2, Square::A3, MoveKind::Quiet);
        test_move_parse(pawn_fen, "a2a3", mv);

        // Pawn double push
        let mv = Move::new(Square::A2, Square::A4, MoveKind::PawnDoublePush);
        test_move_parse(pawn_fen, "a2a4", mv);

        // Pawn capture
        let mv = Move::new(Square::E3, Square::F4, MoveKind::Capture);
        test_move_parse(pawn_fen, "e3f4", mv);

        // Pawn en passant capture
        let mv = Move::new(Square::G5, Square::F6, MoveKind::EnPassantCapture);
        test_move_parse(pawn_fen, "g5f6", mv);

        // Pawn promotion to queen
        let mv = Move::new(
            Square::B7,
            Square::B8,
            MoveKind::promotion(PieceKind::Queen),
        );
        test_move_parse(pawn_fen, "b7b8Q", mv);

        // Pawn promotion to Knight
        let mv = Move::new(
            Square::B7,
            Square::B8,
            MoveKind::promotion(PieceKind::Knight),
        );
        test_move_parse(pawn_fen, "b7b8n", mv);

        // Pawn capture promotion to queen
        let mv = Move::new(
            Square::B7,
            Square::C8,
            MoveKind::promotion_capture(PieceKind::Queen),
        );
        test_move_parse(pawn_fen, "b7c8q", mv);

        // Pawn capture promotion to Knight
        let mv = Move::new(
            Square::B7,
            Square::C8,
            MoveKind::promotion_capture(PieceKind::Knight),
        );
        test_move_parse(pawn_fen, "b7c8N", mv);

        // Now test castling
        let king_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

        // Kingside (short) castling (White)
        let mv = Move::new(Square::E1, Square::G1, MoveKind::ShortCastle);
        test_move_parse(king_fen, "e1g1", mv);

        // Queenside (long) castling (White)
        let mv = Move::new(Square::E1, Square::C1, MoveKind::LongCastle);
        test_move_parse(king_fen, "e1c1", mv);

        // Kingside (short) castling (Black)
        let mv = Move::new(Square::E8, Square::G8, MoveKind::ShortCastle);
        test_move_parse(king_fen, "e8g8", mv);

        // Queenside (long) castling (Black)
        let mv = Move::new(Square::E8, Square::C8, MoveKind::LongCastle);
        test_move_parse(king_fen, "e8c8", mv);
    }
}