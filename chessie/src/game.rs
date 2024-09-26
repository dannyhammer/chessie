/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::{
    fmt::{self, Write},
    ops::Deref,
    str::FromStr,
};

use anyhow::Result;

use crate::{bishop_rays, rook_rays};

use super::{
    bishop_attacks, compute_attacks_by, king_attacks, knight_attacks, pawn_attacks, pawn_pushes,
    queen_attacks, ray_between, ray_containing, rook_attacks, Bitboard, Color, Move, MoveGenIter,
    MoveKind, MoveList, Piece, PieceKind, Position, Rank, Square,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Game {
    /// The current [`Position`] of the game, including piece layouts, castling rights, turn counters, etc.
    position: Position,

    /// All squares whose pieces are attacking the side-to-move's King.
    checkers: Bitboard,

    /// If `self.checkers` is empty, this is [`Bitboard::FULL_BOARD`].
    /// Otherwise, it is the path from every checker to the side-to-move's King.
    /// Because, if we're in check, we must either capture or block the checker.
    checkmask: Bitboard,

    /// All pieces that are the sole blocker between the King and an enemy slider.
    pinned: Bitboard,

    /// All squares (pseudo-legally) attacked by a specific color.
    // attacks_by_color: [Bitboard; Color::COUNT],

    /// Pseudo-legal attacks from every given square on the board.
    // attacks_by_square: [Bitboard; Square::COUNT],

    /// The square where the side-to-move's King resides.
    king_square: Square,
}

impl Game {
    /// Creates a new [`Game`] from  the provided [`Position`].
    pub fn new(position: Position) -> Self {
        /*
        // Compute attack/defend maps by square and color
        let color = position.side_to_move();
        let blockers = position.occupied();

        let mut attacks_by_color = [Bitboard::default(); Color::COUNT];
        let mut attacks_by_square = [Bitboard::default(); Square::COUNT];

        for square in blockers {
            let piece = position.piece_at(square).unwrap();
            let default_attacks = attacks_for(piece, square, blockers);
            attacks_by_square[square] = default_attacks;
            attacks_by_color[color] |= default_attacks;
        }
        */

        let mut game = Self {
            position,
            checkers: Bitboard::default(),
            checkmask: Bitboard::default(),
            pinned: Bitboard::default(),
            // attacks_by_color,
            // attacks_by_square,
            king_square: Square::default(),
        };

        game.recompute_legal_masks();
        game
    }

    /// Creates a new [`Game`] from the provided FEN string.
    #[inline(always)]
    pub fn from_fen(fen: &str) -> Result<Self> {
        Ok(Self::new(Position::from_fen(fen)?))
    }

    /// Copies `self` and returns a [`Game`] after having applied the provided [`Move`].
    #[inline(always)]
    pub fn with_move_made(&self, mv: Move) -> Self {
        let mut copied = *self;
        copied.make_move(mv);
        copied
    }

    /*
    /// Returns `true` if the game is in a position that is identical to a position it has been in before.
    ///
    /// This is useful for checking repetitions.
    ///
    ///
    /// # Example
    /// ```
    /// # use chessie::{Game, Move};
    /// let mut game = Game::default();
    /// game.make_move(Move::from_uci(&game, "b1a3").unwrap());
    /// assert_eq!(game.is_repetition(), false);
    /// game.make_move(Move::from_uci(&game, "b8a6").unwrap());
    /// assert_eq!(game.is_repetition(), false);
    /// game.make_move(Move::from_uci(&game, "a3b1").unwrap());
    /// assert_eq!(game.is_repetition(), false);
    /// game.make_move(Move::from_uci(&game, "a6b8").unwrap());
    /// assert_eq!(game.is_repetition(), true);
    /// ```
    pub fn is_repetition(&self) -> bool {
        for prev in self.history.iter().rev().skip(1).step_by(2) {
            if *prev == self.key() {
                return true;
            }
        }

        false
    }
     */

    /// Applies the move, if it is legal to make. If it is not legal, returns an `Err` explaining why.
    pub fn make_move_checked(&mut self, mv: Move) -> Result<()> {
        self.check_pseudo_legality_of(mv)?;
        self.make_move(mv);
        Ok(())
    }

    /// Recomputes legal metadata (checkers, checkmask, pinmask, etc.).
    #[inline(always)]
    fn recompute_legal_masks(&mut self) {
        let color = self.side_to_move();
        let opponent = color.opponent();
        let occupied = self.occupied();

        self.king_square = self.king(color).to_square_unchecked();

        // Reset the pinmask and checkmask
        self.checkmask = self.enemy_or_empty(color);
        self.pinned = Bitboard::default();

        // Starting off, the easiest checkers to find are Knights and Pawns; just the overlap of their attacks from the King and themselves.
        self.checkers = self.knights(opponent) & knight_attacks(self.king_square)
            | self.pawns(opponent) & pawn_attacks(self.king_square, color);

        // By pretending that there is a Rook/Bishop at our King that can attack without blockers,
        //  we can find all possible sliding attacks *to* the King,
        //  which lets us figure out who our checkers are and what pieces are pinned.
        let enemy_sliding_attacks = rook_rays(self.king_square) & self.orthogonal_sliders(opponent)
            | bishop_rays(self.king_square) & self.diagonal_sliders(opponent);

        // Examine every square that this Rook/Bishop can attack, so that we can figure out if it's a checker or if there are any pinned pieces.
        for attacker in enemy_sliding_attacks {
            // Get a ray between this square and the attacker square, excluding both pieces
            let ray = ray_between(self.king_square, attacker);

            // Whether the piece is a checker or pinned depends on how many pieces are in the ray
            match (ray & occupied).population() {
                // There are no pieces between the attacker and the King, so the attacker is a checker
                0 => self.checkers |= attacker,

                // The piece is not (necessarily) adjacent, but is on a ray to the King, so it is pinned
                1 => self.pinned |= ray & self.color(color), // Enemy pieces can't be pinned!

                // Since we can't move two pieces off of the same ray in the same turn, we don't care about populations higher than 1
                _ => {}
            }
        }

        // If there are any checkers, we need to update the checkmask
        if self.checkers.is_nonempty() {
            // Start with the checkers so they are included in the checkmask (since there is no ray between a King and a Knight)
            self.checkmask = self.checkers;

            // There is *usually* less than two checkers, so this rarely loops.
            for checker in self.checkers {
                self.checkmask |= ray_between(self.king_square, checker);
            }
        }
    }

    /// Applies the provided [`Move`]. No enforcement of legality.
    #[inline(always)]
    pub fn make_move(&mut self, mv: Move) {
        // Actually make the move
        self.position.make_move(mv);

        // Now update movegen metadata
        self.recompute_legal_masks();
    }

    /// Applies the provided [`Move`]s. No enforcement of legality.
    #[inline(always)]
    pub fn make_moves(&mut self, moves: impl IntoIterator<Item = Move>) {
        for mv in moves {
            self.make_move(mv);
        }
    }

    /// Fetch the internal [`Position`] of this [`Game`].
    #[inline(always)]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Fetch a [`Bitboard`] of all squares currently putting the side-to-move's King in check.
    #[inline(always)]
    pub const fn checkers(&self) -> Bitboard {
        self.checkers
    }

    // TODO: Needs testing
    /*
    /// Checks if playing the provided [`Move`] is legal on the current position.
    ///
    /// This aims to be faster than playing the move and recalculating checkmasks and
    /// whatnot by manually moving pieces around and recalculating enemy attacks.
    /// In short, this function (re)moves (captures) appropriate piece(s) (castling),
    /// computes all square attacked by enemy pieces,
    /// and returns whether or not those attacks contain our King.
    pub fn is_legal(&self, mv: Move) -> bool {
        // Create a new board to work with
        let mut board = self.board;

        let from = mv.from();
        let to = mv.to();

        // Remove piece from origin square
        let Some(mut piece) = board.take(from) else {
            return false;
        };
        let color = piece.color();

        // If it's a castle, we need to move the Rook
        if mv.is_castle() {
            // TODO: Chess960
            let castle_index = mv.is_short_castle() as usize;
            let old_rook_square = [Square::A1, Square::H1][castle_index].rank_relative_to(color);
            let new_rook_square = [Square::D1, Square::F1][castle_index].rank_relative_to(color);

            // Move the rook. The King is already handled before and after this if-statement
            let Some(rook) = board.take(old_rook_square) else {
                return false;
            };
            board.place(rook, new_rook_square);
        }

        // If it's en passant, we need to clear the space BEHIND this Pawn
        if mv.is_en_passant() {
            // Safety: En passant can only happen on ranks 3 and 6, so there is guaranteed to be a tile behind `to`
            let ep_square = unsafe { to.backward_by(color, 1).unwrap_unchecked() };
            board.clear(ep_square);
        }

        // Promote the pawn, if applicable
        if let Some(promotion) = mv.promotion() {
            piece = piece.promoted(promotion);
        }

        // Clear destination square and place piece on it
        board.clear(to);
        board.place(piece, to);

        // Compute the enemy attacks to our King
        let king_bb = board.king(color);
        let enemy_attacks =
            compute_attacks_to(&board, king_bb.to_square_unchecked(), color.opponent());

        enemy_attacks.contains(&king_bb)
    }
     */

    /// Generate all legal moves from the current position.
    ///
    /// If you need all moves
    #[inline(always)]
    pub fn get_legal_moves(&self) -> MoveList {
        let mut moves = MoveList::default();
        match self.checkers().population() {
            0 => self.generate_all_moves::<false>(&mut moves),
            1 => self.generate_all_moves::<true>(&mut moves),
            // If we're in double check, we can only move the King
            _ => self.generate_king_moves::<true>(&mut moves),
        }
        moves
    }

    #[inline(always)]
    fn generate_all_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        self.generate_pawn_moves::<IN_CHECK>(moves);
        self.generate_knight_moves::<IN_CHECK>(moves);
        self.generate_bishop_moves::<IN_CHECK>(moves);
        self.generate_rook_moves::<IN_CHECK>(moves);
        self.generate_king_moves::<IN_CHECK>(moves);
    }

    // fn generate_pawn_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
    //     let color = self.side_to_move();

    //     // Any pawn can push forward once, so long as there's nothing blocking it & it's not horizontally pinned
    //     let pawns_that_can_push = self.pawns(color);
    //     let single_pushes = pawns_that_can_push.advance_by(color, 1);
    // }

    /// Creates and appends a [`Move`] that is either a quiet or capture.
    #[inline(always)]
    fn serialize_normal_move(&self, to: Square, from: Square, moves: &mut MoveList) {
        let kind = if self.has(to) {
            MoveKind::Capture
        } else {
            MoveKind::Quiet
        };

        moves.push(Move::new(from, to, kind));
    }

    fn generate_pawn_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        let color = self.side_to_move();
        for from in self.pawns(color) {
            let mobility = self.generate_legal_pawn_mobility::<IN_CHECK>(color, from);

            for to in mobility {
                let mut kind = if self.has(to) {
                    MoveKind::Capture
                } else {
                    MoveKind::Quiet
                };

                if to.rank() == Rank::eighth(color) {
                    // If this move also captures, it's a capture-promote
                    if kind == MoveKind::Capture {
                        moves.push(Move::new(from, to, MoveKind::CaptureAndPromoteKnight));
                        moves.push(Move::new(from, to, MoveKind::CaptureAndPromoteBishop));
                        moves.push(Move::new(from, to, MoveKind::CaptureAndPromoteRook));
                        kind = MoveKind::CaptureAndPromoteQueen;
                    } else {
                        moves.push(Move::new(from, to, MoveKind::PromoteKnight));
                        moves.push(Move::new(from, to, MoveKind::PromoteBishop));
                        moves.push(Move::new(from, to, MoveKind::PromoteRook));
                        kind = MoveKind::PromoteQueen;
                    }
                }
                // If this pawn is moving to the en passant square, it's en passant
                else if Some(to) == self.ep_square() {
                    kind = MoveKind::EnPassantCapture;
                }
                // If the Pawn is moving two ranks, it's a double push
                else if from.rank().abs_diff(to.rank()) == 2 {
                    kind = MoveKind::PawnDoublePush;
                }

                let mv = Move::new(from, to, kind);
                moves.push(mv);
            }
        }
    }

    fn generate_knight_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        let color = self.side_to_move();
        for from in self.knights(color) {
            let attacks = knight_attacks(from);
            let mobility = self.generate_legal_normal_piece_mobility::<IN_CHECK>(from, attacks);

            for to in mobility {
                self.serialize_normal_move(to, from, moves);
            }
        }
    }

    fn generate_bishop_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        let color = self.side_to_move();
        let blockers = self.occupied();
        for from in self.diagonal_sliders(color) {
            let attacks = bishop_attacks(from, blockers);
            let mobility = self.generate_legal_normal_piece_mobility::<IN_CHECK>(from, attacks);

            for to in mobility {
                self.serialize_normal_move(to, from, moves);
            }
        }
    }

    fn generate_rook_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        let color = self.side_to_move();
        let blockers = self.occupied();
        for from in self.orthogonal_sliders(color) {
            let attacks = rook_attacks(from, blockers);
            let mobility = self.generate_legal_normal_piece_mobility::<IN_CHECK>(from, attacks);

            for to in mobility {
                self.serialize_normal_move(to, from, moves);
            }
        }
    }

    fn generate_king_moves<const IN_CHECK: bool>(&self, moves: &mut MoveList) {
        let from = self.king_square;
        let color = self.side_to_move();
        for to in self.generate_legal_king_mobility::<IN_CHECK>(color, from) {
            let mut kind = if self.has(to) {
                MoveKind::Capture
            } else {
                MoveKind::Quiet
            };

            if from == Square::E1.rank_relative_to(color) {
                if to == Square::G1.rank_relative_to(color) {
                    kind = MoveKind::ShortCastle;
                } else if to == Square::C1.rank_relative_to(color) {
                    kind = MoveKind::LongCastle;
                }
            }

            let mv = Move::new(from, to, kind);
            moves.push(mv);
        }
    }

    /*
    /// Generate all legal captures from the current position.
    ///
    /// **Note**: This does not include en passant, for simplicity
    pub fn get_legal_captures(&self) -> MoveList {
        self.iter().only_captures().collect()
    }
     */

    /*
    /// Yields a [`MoveGenIter`] to iterate over all legal moves available in the current position.
    ///
    /// If your intent is to search _every_ available move, use [`Game::get_legal_moves`] instead.
    pub fn iter(&self) -> MoveGenIter {
        MoveGenIter::new(self)
    }
     */

    /*
    // TODO: https://github.com/dannyhammer/brogle/issues/9
    fn compute_pawn_moves(
        &self,
        color: Color,
        checkmask: Bitboard,
        moves: &mut MoveList,
    ) {
        // Fetch all pinned and unpinned pawns
        let pinned_pawns = self.pawns(color) & self.pinmask();
        let unpinned_pawns = self.pawns(color) & !self.pinmask();
        // eprintln!("PINNED PAWNS:\n{pinned_pawns:?}");
        // eprintln!("UNPINNED PAWNS:\n{unpinned_pawns:?}");

        // Pinned pawns may push along their pin ray
        let pinned_pushes = pinned_pawns.advance_by(color, 1) & self.pinmask();
        // Unpinned pawns may push normally
        let unpinned_pushes = unpinned_pawns.advance_by(color, 1);
        let pushes = pinned_pushes | unpinned_pushes;
        // eprintln!("PUSHES:\n{pushes:?}");

        // Cannot push outside of checkmask or into an occupied spot
        let legal_push_mask = !self.occupied() & checkmask;
        let single_pushes = pushes & legal_push_mask;
        // If it can push once, check if it's on the third rank. If so, it can push again.
        let third_rank = Bitboard::third_rank(color);
        let double_pushes = (single_pushes & third_rank).advance_by(color, 1) & legal_push_mask;

        // eprintln!("DOUBLE PUSHES:\n{double_pushes:?}");

        // Cannot capture outside of checkmask or into an empty or friendly spot
        let legal_enemies = self.color(color.opponent()) & checkmask;
        let east_captures = self.pawns(color).advance_by(color, 1).east() & legal_enemies;
        let west_captures = self.pawns(color).advance_by(color, 1).west() & legal_enemies;

        // Now generate the moves for these
        for to in single_pushes {
            let from = to.backward_by(color, 1).unwrap();
            if to.rank() == Rank::eighth(color) {
                moves.push(Move::new(from, to, MoveKind::Promote(PieceKind::Knight)));
                moves.push(Move::new(from, to, MoveKind::Promote(PieceKind::Bishop)));
                moves.push(Move::new(from, to, MoveKind::Promote(PieceKind::Rook)));
                moves.push(Move::new(from, to, MoveKind::Promote(PieceKind::Queen)));
            } else {
                moves.push(Move::new(from, to, MoveKind::Quiet));
            }
        }

        for to in double_pushes {
            let from = to.backward_by(color, 2).unwrap();
            moves.push(Move::new(from, to, MoveKind::Quiet));
        }

        for to in west_captures {
            let from = to.backward_by(color, 1).unwrap().left_by(color, 1).unwrap();

            if to.rank() == Rank::eighth(color) {
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Knight)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Bishop)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Rook)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Queen)));
            } else {
                moves.push(Move::new(from, to, MoveKind::Quiet));
            }
        }

        for to in east_captures {
            let from = to
                .backward_by(color, 1)
                .unwrap()
                .right_by(color, 1)
                .unwrap();
            if to.rank() == Rank::eighth(color) {
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Knight)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Bishop)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Rook)));
                moves.push(Move::new(from, to, MoveKind::PromoCapt(PieceKind::Queen)));
            } else {
                moves.push(Move::new(from, to, MoveKind::Quiet));
            }
        }
    }
     */

    /// Returns `true` if the side-to-move is currently in check.
    #[inline(always)]
    pub const fn is_in_check(&self) -> bool {
        self.checkers.population() > 0
    }

    /// Returns `true` if the side-to-move is currently in double check (in check by more than one piece).
    #[inline(always)]
    pub const fn is_in_double_check(&self) -> bool {
        self.checkers.population() > 1
    }

    /// Generates a [`Bitboard`] of all legal moves for `piece` at `square`.
    pub(crate) fn generate_legal_mobility_for<const IN_CHECK: bool>(
        &self,
        piece: Piece,
        square: Square,
    ) -> Bitboard {
        // Only Pawns and Kings have special movement
        match piece.kind() {
            PieceKind::Pawn => self.generate_legal_pawn_mobility::<IN_CHECK>(piece.color(), square),
            PieceKind::King => self.generate_legal_king_mobility::<IN_CHECK>(piece.color(), square),

            // The remaining pieces all behave the same- just with different default attacks
            PieceKind::Knight => self
                .generate_legal_normal_piece_mobility::<IN_CHECK>(square, knight_attacks(square)),
            PieceKind::Bishop => self.generate_legal_normal_piece_mobility::<IN_CHECK>(
                square,
                bishop_attacks(square, self.occupied()),
            ),
            PieceKind::Rook => self.generate_legal_normal_piece_mobility::<IN_CHECK>(
                square,
                rook_attacks(square, self.occupied()),
            ),
            PieceKind::Queen => self.generate_legal_normal_piece_mobility::<IN_CHECK>(
                square,
                queen_attacks(square, self.occupied()),
            ),
        }
    }

    /// Generates a [`Bitboard`] of all legal moves for a Pawn at `square`.
    fn generate_legal_pawn_mobility<const IN_CHECK: bool>(
        &self,
        color: Color,
        square: Square,
    ) -> Bitboard {
        let blockers = self.occupied();

        // Pinned pawns are complicated:
        // - A pawn pinned horizontally cannot move. At all.
        // - A pawn pinned vertically can only push forward, not capture.
        // - A pawn pinned diagonally can only capture it's pinner.
        let is_pinned = self.pinned.intersects(square);
        let pinmask = Bitboard::from_bool(!is_pinned) | ray_containing(square, self.king_square);

        // If en passant can be performed, check its legality.
        // If not, default to an empty bitboard.
        let ep_bb = self
            .ep_square()
            .map(|ep_square| self.generate_ep_bitboard(color, square, ep_square))
            .unwrap_or_default();

        // Get a mask for all possible pawn double pushes.
        let all_but_this_pawn = blockers ^ square;
        let double_push_mask = all_but_this_pawn | all_but_this_pawn.forward_by(color, 1);
        let pushes = pawn_pushes(square, color) & !double_push_mask & !blockers;

        // Attacks are only possible on enemy occupied squares, or en passant.
        let enemies = self.color(color.opponent());
        let attacks = pawn_attacks(square, color) & (enemies | ep_bb);

        // Pseudo-legal      ---------------Legal--------------
        (pushes | attacks) & (self.checkmask | ep_bb) & pinmask
    }

    /// Generate a [`Bitboard`] for the legality of performing an en passant capture with the Pawn at `square`.
    ///
    /// If en passant is legal, the returned bitboard will have a single bit set, representing a legal capture for the Pawn at `square`.
    /// If en passant is not legal, the returned bitboard will be empty.
    fn generate_ep_bitboard(&self, color: Color, square: Square, ep_square: Square) -> Bitboard {
        // If this Pawn isn't on an adjacent file and the same rank as the enemy Pawn that caused en passant to be possible, it can't perform en passant
        if square.distance_ranks(ep_square) != 1 || square.distance_files(ep_square) != 1 {
            return Bitboard::default();
        }

        // Compute a blockers bitboard as if EP was performed.
        let ep_bb = ep_square.bitboard();
        let ep_target_bb = ep_bb.backward_by(color, 1);
        let blockers_after_ep = (self.occupied() ^ ep_target_bb ^ square) | ep_bb;

        // If, after performing EP, any sliders can attack our King, EP is not legal
        let enemy_ortho_sliders = self.orthogonal_sliders(color.opponent());
        if (rook_attacks(self.king_square, blockers_after_ep) & enemy_ortho_sliders).is_nonempty() {
            return Bitboard::default();
        }

        let enemy_diag_sliders = self.diagonal_sliders(color.opponent());
        if (bishop_attacks(self.king_square, blockers_after_ep) & enemy_diag_sliders).is_nonempty()
        {
            return Bitboard::default();
        }

        // Otherwise, it is safe to perform EP
        ep_bb
    }

    /// Generates a [`Bitboard`] of all legal moves for the King at `square`.
    fn generate_legal_king_mobility<const IN_CHECK: bool>(
        &self,
        color: Color,
        square: Square,
    ) -> Bitboard {
        let attacks = king_attacks(square);
        let enemy_attacks = compute_attacks_by(self, color.opponent());

        // If in check, we cannot castle- we can only attack with the default movement of the King.
        let castling = if IN_CHECK {
            Bitboard::default()
        } else {
            // Otherwise, compute castling availability like normal
            let short = self.castling_rights_for(color).short.map(|rook| {
                self.generate_castling_bitboard(
                    Square::new(rook, Rank::first(color)),
                    Square::G1.rank_relative_to(color),
                    enemy_attacks,
                )
            });

            let long = self.castling_rights_for(color).long.map(|rook| {
                self.generate_castling_bitboard(
                    Square::new(rook, Rank::first(color)),
                    Square::C1.rank_relative_to(color),
                    enemy_attacks,
                )
            });

            short.unwrap_or_default() | long.unwrap_or_default()
        };

        let discoverable_checks = self.generate_discoverable_checks_bitboard(color);

        // Safe squares are ones not attacked by the enemy or part of a discoverable check
        let safe_squares = !(enemy_attacks | discoverable_checks);

        // All legal attacks that are safe and not on friendly squares
        (attacks | castling) & safe_squares & self.enemy_or_empty(color)
    }

    /// Generate a bitboard for `color`'s ability to castle with the Rook on `rook_square`, which will place the King on `dst_square`.
    fn generate_castling_bitboard(
        &self,
        rook_square: Square,
        dst_square: Square,
        enemy_attacks: Bitboard,
    ) -> Bitboard {
        // All squares between the King and Rook must be empty
        let blockers = self.occupied();
        let squares_that_must_be_empty = ray_between(self.king_square, rook_square);
        let squares_are_empty = (squares_that_must_be_empty & blockers).is_empty();

        // All squares between the King and his destination must not be attacked
        let squares_that_must_be_safe = ray_between(self.king_square, dst_square);
        let squares_are_safe = (squares_that_must_be_safe & enemy_attacks).is_empty();

        Bitboard::from_square(dst_square)
            & Bitboard::from_bool(squares_are_empty && squares_are_safe)
    }

    /// These are the rays containing the King and his Checkers.
    /// They are used to prevent the King from retreating along a line he is checked on.
    /// Note: A pawn can't generate a discoverable check, as it can only capture 1 square away.
    #[inline(always)]
    fn generate_discoverable_checks_bitboard(&self, color: Color) -> Bitboard {
        let mut discoverable = Bitboard::default();

        for checker in self.checkers & self.sliders(color.opponent()) {
            // Need to XOR because capturing the checker is legal
            discoverable |= ray_containing(self.king_square, checker) ^ checker;
        }

        discoverable
    }

    /// Generates a [`Bitboard`] of all legal moves for a non-Pawn and non-King piece at `square`.
    #[inline(always)]
    fn generate_legal_normal_piece_mobility<const IN_CHECK: bool>(
        &self,
        square: Square,
        default_attacks: Bitboard,
    ) -> Bitboard {
        // Check if this piece is pinned along any of the pinmasks
        let is_pinned = self.pinned.intersects(square);
        let pinmask = Bitboard::from_bool(!is_pinned) | ray_containing(square, self.king_square);

        // Pseudo-legal attacks that are within the check/pin mask and attack non-friendly squares
        default_attacks & self.checkmask & pinmask
    }
}

impl Deref for Game {
    type Target = Position;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.position
    }
}

impl FromStr for Game {
    type Err = anyhow::Error;
    #[inline(always)]
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_fen(s)
    }
}

impl Default for Game {
    #[inline(always)]
    fn default() -> Self {
        Self::new(Position::default())
    }
}

impl<'a> IntoIterator for &'a Game {
    type IntoIter = MoveGenIter<'a>;
    type Item = Move;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        MoveGenIter::new(self)
    }
}

impl<'a> IntoIterator for &'a mut Game {
    type IntoIter = MoveGenIter<'a>;
    type Item = Move;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        MoveGenIter::new(self)
    }
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.position().fmt(f)
    }
}

impl fmt::Debug for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = |to_fmt: &[(Bitboard, &str)]| {
            let strings = to_fmt
                .iter()
                .map(|(b, s)| (b.to_string(), s))
                .collect::<Vec<_>>();

            let splits = strings
                .iter()
                .map(|(b, _)| b.split('\n').collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let labels = strings.iter().fold(String::new(), |mut acc, (_, s)| {
                _ = write!(acc, "{s:10}\t\t");
                acc
            });

            let boards = (0..8).fold(String::new(), |mut acc, i| {
                _ = writeln!(
                    acc,
                    "{}",
                    (0..splits.len()).fold(String::new(), |mut output, j| {
                        _ = write!(output, "{}\t", splits[j][i]);
                        output
                    })
                );
                acc
            });

            format!("{labels}\n{boards}")
        };

        let color = self.position.side_to_move();

        let check_data = format(&[
            (self.checkers, "Checkers"),
            (self.checkmask, "Checkmask"),
            (self.pinned, "Pinned"),
            (
                self.generate_discoverable_checks_bitboard(color),
                "Disc. Checks",
            ),
        ]);

        let mobility_data = format(&[
            (self.position.color(color), "Friendlies"),
            (self.position.color(color.opponent()), "Enemies"),
        ]);

        write!(
            f,
            "Position:\n{:?}\n\n{check_data}\n\n{mobility_data}",
            self.position
        )
    }
}
