/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use super::{
    Bitboard, Board, Color, Game, Move, MoveKind, Piece, PieceKind, Rank, Square, MAX_NUM_MOVES,
};

// Include the pre-generated magics
include!("magics.rs");
// include!(concat!(env!("OUT_DIR"), "/rook_magics.rs"));

/// An alias for an [`arrayvec::ArrayVec`] containing at most [`MAX_NUM_MOVES`] moves.
pub type MoveList = arrayvec::ArrayVec<Move, MAX_NUM_MOVES>;

const RAY_BETWEEN: [[Bitboard; Square::COUNT]; Square::COUNT] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/ray_between.dat"
    )))
};

const RAY_CONTAINING: [[Bitboard; Square::COUNT]; Square::COUNT] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/ray_containing.dat"
    )))
};

const KNIGHT_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/knight_attacks.dat"
    )))
};

const ROOK_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/rook_attacks.dat"
    )))
};

const BISHOP_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/bishop_attacks.dat"
    )))
};

const KING_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/king_attacks.dat"
    )))
};

const WHITE_PAWN_PUSHES: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/white_pawn_pushes.dat"
    )))
};

const BLACK_PAWN_PUSHES: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/black_pawn_pushes.dat"
    )))
};

const WHITE_PAWN_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/white_pawn_attacks.dat"
    )))
};

const BLACK_PAWN_ATTACKS: [Bitboard; 64] = unsafe {
    std::mem::transmute(*include_bytes!(concat!(
        env!("OUT_DIR"),
        "/black_pawn_attacks.dat"
    )))
};

/// An iterator over all legal moves in a specific position.
///
/// This is slower than calling [`Game::get_legal_moves`], but provides more flexibility (such as only generating captures).
pub struct MoveGenIter<'a> {
    /// The [`Game`] to generate moves for.
    game: &'a Game,

    /// The move generator will only generate moves that place pieces on squares defined in this mask.
    ///
    /// By default, this is set to empty and enemy-occupied squares.
    to_mask: Bitboard,

    /// The move generator will only generate moves that moves pieces from squares defined in this mask.
    ///
    /// By default, this is set to friendly-occupied squares.
    /// However, if the King is in double-check, this is set to a bitboard of only the King's square,
    /// and cannot be overridden.
    from_mask: Bitboard,

    /// Current [`Square`] that we're generating moves for.
    square: Square,

    /// Current mobility of the [`Piece`] at `self.square`.
    mobility: Bitboard,

    /// Current promotion index, for enumerating Pawn moves.
    promotion: u8,
}

impl<'a> MoveGenIter<'a> {
    /// Create a new [`MoveGenIter`] from the provided [`Game`].
    pub fn new(game: &'a Game) -> Self {
        let color = game.side_to_move();

        // Can only go to squares that are empty or enemy-occupied
        let to_mask = game.enemy_or_empty(color);

        // If we're in double check, we can only generate moves for the King
        let from_mask = if game.is_in_double_check() {
            game.king(color)
        } else {
            game.color(color)
        };

        Self {
            square: from_mask.lsb().unwrap_or_default(),
            from_mask,
            to_mask,
            mobility: Bitboard::EMPTY_BOARD,
            promotion: 0,
            game,
        }
    }

    /// Consumes `self`, returning a [`MoveGenIter`] that only generates moves _from_ the squares in `mask`.
    ///
    /// Note that if the position is currently in double check, the provided mask will have no effect,
    /// as only the King is allowed to move during a double check.
    #[inline(always)]
    pub fn only_moves_from(mut self, mask: impl Into<Bitboard>) -> Self {
        // If in double check, we cannot generate moves for anything other than the King
        self.from_mask = if self.game.is_in_double_check() {
            self.game.king(self.game.side_to_move())
        } else {
            mask.into()
        };

        self
    }

    /// Consumes `self`, returning a [`MoveGenIter`] that only generates moves _to_ the squares in `mask`.
    #[inline(always)]
    pub fn only_moves_to(mut self, mask: impl Into<Bitboard>) -> Self {
        self.to_mask = mask.into();
        self
    }

    /// Consumes `self`, returning a [`MoveGenIter`] that only generates moves that capture enemy pieces.
    ///
    /// **Note**: This does not include en passant, for simplicity
    #[inline(always)]
    pub fn only_captures(self) -> Self {
        let opponent = self.game.side_to_move().opponent();
        let mask = self.game.color(opponent);
        self.only_moves_to(mask)
    }

    /// Constructs a [`Move`], given the `from` and `to` squares, as well as the struct's internal context.
    fn serialize_move(&mut self, from: Square, to: Square) -> Move {
        let piece = self.game.piece_at_unchecked(from); // Safe because there must be a piece here for us to move.

        let color = self.game.side_to_move();

        // By default, the move is either quiet or a capture
        let mut kind = if self.game.has(to) {
            MoveKind::Capture
        } else {
            MoveKind::Quiet
        };

        if piece.is_pawn() {
            // If a promotion was provided, it's a promotion of some kind
            if to.rank() == Rank::eighth(color) {
                // Fetch the kind of piece to which this Pawn will be promoted.
                // We add 1 here to shift our range to exclude the Pawn.
                let promotion = PieceKind::from_bits_unchecked(self.promotion + 1);

                // Increment the promotion index, resetting to 4 if it reaches 4.
                self.promotion = (self.promotion + 1) & 3;

                // If this move also captures, it's a capture-promote
                if kind == MoveKind::Capture {
                    kind = MoveKind::promotion_capture(promotion);
                } else {
                    kind = MoveKind::promotion(promotion);
                }
            }
            // If this pawn is moving to the en passant square, it's en passant
            else if Some(to) == self.game.ep_square() {
                kind = MoveKind::EnPassantCapture;
            }
            // If the Pawn is moving two ranks, it's a double push
            else if from.rank().abs_diff(to.rank()) == 2 {
                kind = MoveKind::PawnDoublePush;
            }
        } else if piece.is_king() && from == Square::E1.rank_relative_to(color) {
            if to == Square::G1.rank_relative_to(color) {
                kind = MoveKind::ShortCastle;
            } else if to == Square::C1.rank_relative_to(color) {
                kind = MoveKind::LongCastle;
            }
        }

        Move::new(from, to, kind)
    }

    /// Progresses the iterator and yields the next [`Move`] available, if any remain.
    fn get_next_move(&mut self) -> Option<Move> {
        // If we've enumerated all available moves for the current piece, move on to the next piece in the "from" mask.
        // We loop here because the piece at the next square in the "from" mask may not have any legal moves, so we must skip it.
        while self.mobility.is_empty() {
            self.square = self.from_mask.pop_lsb()?;
            let piece = self.game.piece_at_unchecked(self.square); // Square is guaranteed to have piece on it.

            // Mask off anything not in the "to" mask
            self.mobility = if self.game.is_in_check() {
                self.game
                    .generate_legal_mobility_for::<true>(piece, self.square)
            } else {
                self.game
                    .generate_legal_mobility_for::<false>(piece, self.square)
            } & self.to_mask;
        }

        // Create a `Move` struct from the move data.
        let to = self.mobility.lsb_unchecked(); // Guaranteed to be at least one destination square, from the loop above.
        let mv = self.serialize_move(self.square, to);

        // If a Pawn promotion occurred, we may need to enumerate more promotions
        if self.promotion == 0 {
            self.mobility.clear_lsb();
        }

        Some(mv)
    }
}

impl<'a> Iterator for MoveGenIter<'a> {
    type Item = Move;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_move()
    }
}

/// Computes a checkmask for the current board state.
///
/// The "checkmask" determines what squares we can legally move *to*.
/// If there are no checkers, the checkmask contains any square that is occupied by the enemy or is empty.
/// Otherwise, the checkmask is the path from the checker(s) to the King (`square`),
/// because we must either block the check or capture the checker.
#[inline(always)]
pub fn compute_checkmask(
    board: &Board,
    checkers: Bitboard,
    square: Square,
    color: Color,
) -> Bitboard {
    if checkers.is_empty() {
        board.enemy_or_empty(color)
    } else {
        // Start with the checkers so they are included in the checkmask (since there is no ray between a King and a Knight)
        let mut checkmask = checkers;

        // There is *usually* only one checker, so this rarely loops.
        for checker in checkers {
            checkmask |= ray_between(square, checker);
        }

        checkmask
    }
}

/// Computes a [`Bitboard`] of all the `color` pieces that attack `square`.
///
/// This _only_ includes the attacker itself, not the ray of attack.
/// It is useful for finding checkers.
#[inline(always)]
pub fn compute_attackers_to(board: &Board, square: Square, color: Color) -> Bitboard {
    let occupied = board.occupied();

    (pawn_attacks(square, color.opponent()) & board.pawns(color))
        | (knight_attacks(square) & board.knights(color))
        | (bishop_attacks(square, occupied) & board.diagonal_sliders(color))
        | (rook_attacks(square, occupied) & board.orthogonal_sliders(color))
        | (king_attacks(square) & board.king(color))
}

/// Computes a [`Bitboard`] of all squares attacked by `color`.
///
/// This is a raw attack map, meaning the squares on this map are unsafe for the friendly King to occupy.
#[inline(always)]
pub fn compute_attacks_by(board: &Board, color: Color) -> Bitboard {
    let blockers = board.occupied();

    let mut attacks = pawn_attack_map(board, color);
    for square in board.knights(color) {
        attacks |= knight_attacks(square);
    }
    for square in board.diagonal_sliders(color) {
        attacks |= bishop_attacks(square, blockers);
    }
    for square in board.orthogonal_sliders(color) {
        attacks |= rook_attacks(square, blockers);
    }
    for square in board.king(color) {
        attacks |= king_attacks(square);
    }

    attacks
}

/// Computes a [`Bitboard`] of all squares attacked by `color` Pawns, excluding En Passant for convenience.
#[inline(always)]
pub fn pawn_attack_map(board: &Board, color: Color) -> Bitboard {
    let pushes = board.pawns(color).forward_by(color, 1);
    pushes.east() | pushes.west()
}

/*
/// Computes a [`Bitboard`] of all squares that `color` Pawns can push to.
///
/// These are *not* capture moves. Only forward pushes and double-pushes.
#[inline(always)]
pub fn pawn_push_map(board: &Board, color: Color) -> Bitboard {
    let pawns = board.pawns(color);
    let empty = board.empty();

    // By default, all pawns can push forward once if they are not blocked
    let single_pushes = pawns.forward_by(color, 1) & empty;

    // Any pawn that started on the 2nd rank and was able to push once can also push again, if unblocked
    let double_pushes = (Bitboard::third_rank(color) & single_pushes).forward_by(color, 1) & empty;

    single_pushes | double_pushes
}
 */

pub fn compute_checkers_and_pinmask(
    board: &Board,
    square: Square,
    color: Color,
) -> (Bitboard, Bitboard) {
    // let checkers = compute_attackers_to(board, square, color);
    let mut checkers = Bitboard::EMPTY_BOARD;
    let mut pinmask = Bitboard::EMPTY_BOARD;

    let opponent = color.opponent();
    let occupied = board.occupied();

    // By treating this square like a rook/bishop that can attack "through" anything, we can find all of the possible attacks *to* this square by these enemy pieces, including possible pins
    let sliding_attacks = ROOK_ATTACKS[square] & board.orthogonal_sliders(opponent)
        | BISHOP_ATTACKS[square] & board.diagonal_sliders(opponent);

    // If a slider is reachable from this square, then it is attacking this square
    for attacker_square in sliding_attacks {
        // Get a ray between this square and the attacker square, excluding both pieces
        let pieces_between = ray_between(square, attacker_square) & occupied;

        // Whether the piece is a checker or pinned depends on how many pieces are in the ray
        match pieces_between.population() {
            // Checker
            0 => checkers |= attacker_square,
            // Pinned piece
            1 => pinmask |= pieces_between,
            // Don't care
            _ => {}
        }
    }

    checkers |= board.knights(opponent) & knight_attacks(square);
    checkers |= board.pawns(opponent) & pawn_attacks(square, color);

    (checkers, pinmask)
}

/// Computes a pinmask of the board, centered on `square`.
///
/// A pinmask contains the orthogonal and diagonal rays containing `square` and any enemy sliders,
/// if and only if there is only 1 piece between `square` and the enemy slider.
///
/// When `square` is the King's square, any pieces within the pinmask must not move outside of the pinmask.
pub fn compute_pinmask_for(board: &Board, square: Square, color: Color) -> Bitboard {
    let mut pinmask = Bitboard::EMPTY_BOARD;
    let opponent = color.opponent();
    let occupied = board.occupied();

    // By treating this square like a rook/bishop that can attack "through" anything, we can find all of the possible attacks *to* this square by these enemy pieces, including possible pins
    let sliding_attacks = ROOK_ATTACKS[square] & board.orthogonal_sliders(opponent)
        | BISHOP_ATTACKS[square] & board.diagonal_sliders(opponent);

    // If a slider is reachable from this square, then it is attacking this square
    for attacker_square in sliding_attacks {
        // Get a ray between this square and the attacker square, excluding both pieces
        let pinned_pieces = ray_between(square, attacker_square) & occupied;

        // A ray is a pin if there is only one piece along it
        if pinned_pieces.population() == 1 {
            pinmask |= pinned_pieces;
        }
    }

    pinmask
}

/// Fetch the default, pseudo-legal attacks for `piece` at `square`, given `blockers`.
///
/// Note: For Pawns, this retrieves only the Pawn's _attacks_, not their pushes.
/// This is because we call this function internally when generating legal moves to create
/// an "attack map" of all attacks possible, indexable by a [`Square`].
/// This map is then used to determine if a square is "safe" for the King to move to.
/// If Pawn pushes were included in this "attack map", then the King would not be able to
/// move to squares otherwise be safe, as the move generator would think that a Pawn's
/// threat of pushing could check the King.
#[inline(always)]
pub const fn attacks_for(piece: Piece, square: Square, blockers: Bitboard) -> Bitboard {
    match piece.kind() {
        PieceKind::Pawn => pawn_attacks(square, piece.color()),
        PieceKind::Knight => knight_attacks(square),
        PieceKind::Bishop => bishop_attacks(square, blockers),
        PieceKind::Rook => rook_attacks(square, blockers),
        PieceKind::Queen => queen_attacks(square, blockers),
        PieceKind::King => king_attacks(square),
    }
}

/// Fetches a [`Bitboard`] with all of the bits along the ray between `from` and `to` (exclusive) set to `1`.
///
/// # Example
/// ```
/// # use chessie::*;
/// assert_eq!(ray_between(Square::A1, Square::A8), Bitboard::FILE_A ^ Square::A1 ^ Square::A8);
/// ```
#[inline(always)]
pub const fn ray_between(from: Square, to: Square) -> Bitboard {
    RAY_BETWEEN[from.index()][to.index()]
}

/// Fetches a [`Bitboard`] with all of the bits along the ray containing `from` and `to` set to `1`.
///
/// # Example
/// ```
/// # use chessie::*;
/// assert_eq!(ray_containing(Square::A3, Square::A5), Bitboard::FILE_A);
/// ```
#[inline(always)]
pub const fn ray_containing(from: Square, to: Square) -> Bitboard {
    RAY_CONTAINING[from.index()][to.index()]
}

/// Computes the possible moves for a Rook at a given [`Square`] with the provided blockers.
///
/// This will yield a [`Bitboard`] that allows the Rook to capture the first blocker.
#[inline(always)]
pub const fn rook_attacks(square: Square, blockers: Bitboard) -> Bitboard {
    Bitboard::new(ROOK_MOVES[magic_index(&ROOK_MAGICS[square.index()], blockers)])
}

/// Computes the (unblocked) moves for a Rook at a given [`Square`].
#[inline(always)]
pub const fn rook_rays(square: Square) -> Bitboard {
    ROOK_ATTACKS[square.index()]
}

/// Computes the possible moves for a Bishop at a given [`Square`] with the provided blockers.
///
/// This will yield a [`Bitboard`] that allows the Bishop to capture the first blocker.
#[inline(always)]
pub const fn bishop_attacks(square: Square, blockers: Bitboard) -> Bitboard {
    Bitboard::new(BISHOP_MOVES[magic_index(&BISHOP_MAGICS[square.index()], blockers)])
}

/// Computes the (unblocked) moves for a Bishop at a given [`Square`].
#[inline(always)]
pub const fn bishop_rays(square: Square) -> Bitboard {
    BISHOP_ATTACKS[square.index()]
}

/// Computes the possible moves for a Queen at a given [`Square`] with the provided blockers.
///
/// This will yield a [`Bitboard`] that allows the Queen to capture the first blocker.
#[inline(always)]
pub const fn queen_attacks(square: Square, blockers: Bitboard) -> Bitboard {
    rook_attacks(square, blockers).or(bishop_attacks(square, blockers))
}

/// Fetch the raw, unblocked attacks for a knight on the provided square.
#[inline(always)]
pub const fn knight_attacks(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index()]
}

/// Fetch the raw, unblocked attacks for a king on the provided square.
#[inline(always)]
pub const fn king_attacks(square: Square) -> Bitboard {
    KING_ATTACKS[square.index()]
}

/// Computes a [`Bitboard`] for the attacks and pushes of a Pawn at `square`.
///
/// If a Pawn is on it's second rank, it can push two (only if unblocked).
/// If there is a blocker diagonal of the Pawn, it can be captured.
pub const fn pawn_moves(square: Square, color: Color, blockers: Bitboard) -> Bitboard {
    // By default, a pawn can push forward two on it's starting rank, and one elsewhere
    // We get a mask of all the blockers (minus this piece) and shift it twice forward
    // So, if this pawn *could* move forward twice, but there was a piece directly in front of it, it now cannot move
    let all_but_this_pawn = blockers.xor(square.bitboard());
    let shift_mask = all_but_this_pawn.or(all_but_this_pawn.forward_by(color, 1));

    // If there is a piece in front of this pawn, we cannot push two
    let pushes = pawn_pushes(square, color).and(shift_mask.not());

    // By default, a pawn can only attack if there is a piece available to take
    let attacks = pawn_attacks(square, color).and(blockers);

    pushes.or(attacks)
}

// const PAWN_ATTACKS: [&'static [Bitboard; 64]; 2] = [&WHITE_PAWN_ATTACKS, &BLACK_PAWN_ATTACKS];
// const PAWN_PUSHES: [&'static [Bitboard; 64]; 2] = [&WHITE_PAWN_PUSHES, &BLACK_PAWN_PUSHES];

/// Fetch the raw, unblocked pushes for a pawn of the provided color on the provided square.
#[inline(always)]
pub const fn pawn_pushes(square: Square, color: Color) -> Bitboard {
    match color {
        Color::White => WHITE_PAWN_PUSHES[square.index()],
        Color::Black => BLACK_PAWN_PUSHES[square.index()],
    }
}

/// Fetch the raw, unblocked attacks for a pawn of the provided color on the provided square.
#[inline(always)]
pub const fn pawn_attacks(square: Square, color: Color) -> Bitboard {
    match color {
        Color::White => WHITE_PAWN_ATTACKS[square.index()],
        Color::Black => BLACK_PAWN_ATTACKS[square.index()],
    }
}

struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u8,
    offset: u32,
}

#[inline(always)]
const fn magic_index(entry: &MagicEntry, blockers: Bitboard) -> usize {
    let blockers = blockers.inner() & entry.mask;
    let hash = blockers.wrapping_mul(entry.magic);
    let index = (hash >> entry.shift) as usize;
    entry.offset as usize + index
}

#[cfg(test)]
mod test {
    use super::*;

    /// Checks if `moves` and `legal_moves` contain all the same elements, ignoring order
    fn lists_match(moves: Bitboard, legal_moves: &[Square]) {
        assert_eq!(
            moves.population() as usize,
            legal_moves.len(),
            "\nMoves: {:?}\nLegal: {:?}",
            moves.iter().collect::<Vec<_>>(),
            legal_moves
        );

        for mv in moves {
            assert!(
                legal_moves.contains(&mv),
                "{} not found in {:?}",
                mv,
                legal_moves
            );
        }
    }

    #[test]
    fn rook_blockers() {
        let legal_moves = [
            Square::D2,
            Square::D3,
            Square::D5,
            Square::D6,
            Square::A4,
            Square::B4,
            Square::C4,
            Square::E4,
            Square::F4,
            Square::G4,
            Square::H4,
        ];

        // . . . X . . . X
        // . . . . . . . .
        // . . . X . . . .
        // . . . . . . . .
        // . . . . . . . X
        // . . X . . . . .
        // . . . X . X . .
        // . . . . . . . .
        let blockers =
            Bitboard::new(0b1000100000000000000010000000000010000000000001000010100000000000);

        let moves = rook_attacks(Square::D4, blockers);

        lists_match(moves, &legal_moves);
    }
}
