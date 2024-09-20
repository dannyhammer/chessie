use std::{
    fmt::{self, Write},
    ops::Deref,
    str::FromStr,
};

use anyhow::Result;

use super::{
    attacks_for, bishop_attacks, compute_attacks_to, compute_pinmask_for, king_attacks,
    knight_attacks, pawn_attacks, pawn_pushes, queen_attacks, ray_between, ray_containing,
    rook_attacks, Bitboard, Color, Move, MoveGenIter, MoveList, Piece, PieceKind, Position, Square,
    ZobristKey,
};

#[derive(Clone, PartialEq, Eq)]
pub struct Game {
    /// The current [`Position`] of the game, including piece layouts, castling rights, turn counters, etc.
    position: Position,

    /// A history of hashed positions, used to detect repetitions and transpositions.
    history: Vec<ZobristKey>,

    /// All squares whose pieces are attacking the side-to-move's King.
    checkers: Bitboard,

    /// If `self.checkers` is empty, this is [`Bitboard::FULL_BOARD`].
    /// Otherwise, it is the path from every checker to the side-to-move's King.
    checkmask: Bitboard,

    /// All rays from enemy sliders to the side-to-move's King where there is only a single piece preventing check.
    pinmask: Bitboard,

    /// If the side-to-move is in Check, this contains the rays between the enemy sliding checkers and the side-to-move's King.
    ///
    /// This is used to prevent the King from "retreating" along the ray that the enemy slider attacks.
    discoverable_checks: Bitboard,

    /// All squares (pseudo-legally) attacked by a specific color.
    // attacks_by_color: [Bitboard; Color::COUNT],

    /// Pseudo-legal attacks from every given square on the board.
    // attacks_by_square: [Bitboard; Square::COUNT],

    /// The square where the side-to-move's King resides.
    king_square: Square,
}

impl Game {
    /// Creates a new, empty [`Game`].
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
            pinmask: Bitboard::default(),
            discoverable_checks: Bitboard::default(),
            // attacks_by_color,
            // attacks_by_square,
            king_square: Square::default(),
            history: Vec::with_capacity(128), // Seems like a good upper bound
        };
        game.update_legal_metadata();
        game
    }

    /// Creates a new [`Game`] from the provided FEN string.
    pub fn from_fen(fen: &str) -> Result<Self> {
        Ok(Self::new(Position::from_fen(fen)?))
    }

    /// Clones `self` and returns a [`Game`] after having applied the provided [`Move`].
    pub fn with_move_made(&self, mv: Move) -> Self {
        // Self::new(self.position.with_move_made(mv))
        let mut cloned = self.clone();
        cloned.make_move(mv);
        cloned
    }

    /// Returns `true` if the game is in a position that is identical to the position it was in before.
    ///
    /// This is for checking "two-fold" repetition.
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

    /// Applies the move, if it is legal to make. If it is not legal, returns an `Err` explaining why.
    pub fn make_move_checked(&mut self, mv: Move) -> Result<()> {
        self.check_legality_of(mv)?;
        self.make_move(mv);
        Ok(())
    }

    /// Applies the provided [`Move`]. No enforcement of legality.
    pub fn make_move(&mut self, mv: Move) {
        // Add the move to the game's history
        self.history.push(self.key());

        // Actually make the move
        self.position.make_move(mv);
        // eprintln!("Pos after {mv}:\n{:?}", self.position);

        /*
        let (from, to, _) = mv.parts();
        // Since the move has already been applied, we must fetch the value at `to`.
        let Some(color) = self.color_at(to) else {
            panic!(
                "Failed to apply {mv:?} to {}: No color found at {from}",
                self.to_fen()
            );
        };
        */

        // Update checkmasks, checkers, pinmasks, etc.
        self.update_legal_metadata();
    }

    /// Applies the provided [`Move`]s. No enforcement of legality.
    pub fn make_moves(&mut self, moves: impl IntoIterator<Item = Move>) {
        for mv in moves {
            self.make_move(mv);
        }
    }

    /// Fetch the history of this [`Game`].
    pub fn history(&self) -> &[ZobristKey] {
        &self.history
    }

    /// Fetch the internal [`Position`] of this [`Game`].
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Fetch a [`Bitboard`] of all squares currently putting the side-to-move's King in check.
    pub const fn checkers(&self) -> Bitboard {
        self.checkers
    }

    /// Generate all legal moves from the current position.
    pub fn get_legal_moves(&self) -> MoveList {
        self.iter().collect()
    }

    /// Generate all legal captures from the current position.
    pub fn get_legal_captures(&self) -> MoveList {
        self.iter().only_captures().collect()
    }

    /// Yields a [`MoveGenIter`] to iterate over all legal moves available in the current position.
    ///
    /// If your intent is to search _every_ available move, use [`Game::get_legal_moves`] instead.
    pub fn iter(&self) -> MoveGenIter {
        MoveGenIter::new(self)
    }

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
    pub const fn is_in_check(&self) -> bool {
        self.checkers.population() > 0
    }

    /// Returns `true` if the side-to-move is currently in double check (in check by more than one piece).
    pub const fn is_in_double_check(&self) -> bool {
        self.checkers.population() > 1
    }

    pub fn compute_attacks_by(&self, color: Color) -> Bitboard {
        let mut attacks = Bitboard::default();
        let blockers = self.occupied();
        for (square, piece) in self.all_for(color) {
            attacks |= attacks_for(piece, square, blockers);
        }
        attacks
    }

    /// Generates a [`Bitboard`] of all legal moves for `piece` at `square`.
    pub(crate) fn generate_legal_mobility_for(&self, piece: Piece, square: Square) -> Bitboard {
        // Only Pawns and Kings have special movement
        match piece.kind() {
            PieceKind::Pawn => self.generate_legal_pawn_mobility(piece.color(), square),
            PieceKind::King => self.generate_legal_king_mobility(piece.color(), square),

            // The remaining pieces all behave the same- just with different default attacks
            PieceKind::Knight => {
                self.generate_legal_normal_piece_mobility(square, knight_attacks(square))
            }
            PieceKind::Bishop => self.generate_legal_normal_piece_mobility(
                square,
                bishop_attacks(square, self.occupied()),
            ),
            PieceKind::Rook => self.generate_legal_normal_piece_mobility(
                square,
                rook_attacks(square, self.occupied()),
            ),
            PieceKind::Queen => self.generate_legal_normal_piece_mobility(
                square,
                queen_attacks(square, self.occupied()),
            ),
        }
    }

    /// Generates a [`Bitboard`] of all legal moves for a Pawn at `square`.
    fn generate_legal_pawn_mobility(&self, color: Color, square: Square) -> Bitboard {
        let blockers = self.occupied();

        // Pinned pawns are complicated:
        // - A pawn pinned horizontally cannot move. At all.
        // - A pawn pinned vertically can only push forward, not capture.
        // - A pawn pinned diagonally can only capture it's pinner.
        let is_pinned = self.pinmask.get(square);
        let pinmask = Bitboard::from_bool(!is_pinned) | ray_containing(square, self.king_square);

        // If en passant can be performed, check its legality.
        // If not, default to an empty bitboard.
        let ep_bb = self
            .ep_square()
            .map(|ep_square| self.generate_ep_bitboard(color, square, ep_square))
            .unwrap_or_default();

        // Get a mask for all possible pawn double pushes.
        let all_but_this_pawn = blockers ^ square;
        let double_push_mask = all_but_this_pawn | all_but_this_pawn.advance_by(color, 1);
        let pushes = pawn_pushes(square, color) & !double_push_mask & !blockers;

        // Attacks are only possible on enemy occupied squares, or en passant.
        let enemies = self.color(color.opponent());
        let attacks = pawn_attacks(square, color) & (enemies | ep_bb);

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
        let ep_target_bb = ep_bb.retreat_by(color, 1);
        let blockers_after_ep = (self.occupied() ^ ep_target_bb ^ square) | ep_bb;

        // If, after performing EP, any orthogonal sliders can attack our King, EP is not legal
        let enemy_ortho_sliders = self.orthogonal_sliders(color.opponent());
        let enemy_diag_sliders = self.diagonal_sliders(color.opponent());

        // If enemy sliders can attack our King, then EP is not legal to perform
        let possible_checkers = (rook_attacks(self.king_square, blockers_after_ep)
            & enemy_ortho_sliders)
            | (bishop_attacks(self.king_square, blockers_after_ep) & enemy_diag_sliders);

        // eprintln!("EP for {color} Pawn at {square}->{ep_square} is safe!");

        Bitboard::from_bool(possible_checkers.is_empty()) & ep_bb
    }

    /// Generates a [`Bitboard`] of all legal moves for the King at `square`.
    fn generate_legal_king_mobility(&self, color: Color, square: Square) -> Bitboard {
        let attacks = king_attacks(square);
        let enemy_attacks = self.compute_attacks_by(color.opponent());

        // If in check, we cannot castle- we can only attack with the default movement of the King.
        let castling = if self.is_in_check() {
            Bitboard::default()
        } else {
            // Otherwise, compute castling availability like normal
            let short = self.castling_rights().short[color].map(|rook_sq| {
                self.generate_castling_bitboard(
                    rook_sq,
                    Square::G1.rank_relative_to(color),
                    enemy_attacks,
                )
            });

            let long = self.castling_rights().long[color].map(|rook_sq| {
                self.generate_castling_bitboard(
                    rook_sq,
                    Square::C1.rank_relative_to(color),
                    enemy_attacks,
                )
            });

            short.unwrap_or_default() | long.unwrap_or_default()
        };

        // Safe squares are ones not attacked by the enemy or part of a discoverable check
        let safe_squares = !(enemy_attacks | self.discoverable_checks); // Not attacked by the enemy (even after King retreats)

        // All legal attacks that are safe and not on friendly squares
        (attacks | castling) & safe_squares
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

    /// Generates a [`Bitboard`] of all legal moves for a non-Pawn and non-King piece at `square`.
    fn generate_legal_normal_piece_mobility(
        &self,
        square: Square,
        default_attacks: Bitboard,
    ) -> Bitboard {
        // Check if this piece is pinned along any of the pinmasks
        let is_pinned = self.pinmask.get(square);
        let pinmask = Bitboard::from_bool(!is_pinned) | ray_containing(square, self.king_square);

        // Pseudo-legal attacks that are within the check/pin mask and attack non-friendly squares
        default_attacks & self.checkmask & pinmask
    }

    /// Updates the legal metadata of the game.
    ///
    /// "Legal metadata" refers to data about which pieces are pinned, whether the King is in check (and by who), and others.
    fn update_legal_metadata(&mut self) {
        let color = self.side_to_move();
        self.king_square = self.king(color).to_square_unchecked();

        // Checkmask and pinmasks for legal move generation
        self.discoverable_checks = Bitboard::EMPTY_BOARD;
        self.checkers = compute_attacks_to(self, self.king_square, color.opponent());
        self.pinmask = compute_pinmask_for(self, self.king_square, color);

        // These are the rays containing the King and his Checkers
        // They are used to prevent the King from retreating along a line he is checked on!
        // Note: A pawn can't generate a discoverable check, as it can only capture 1 square away.
        for checker in self.checkers & self.sliders(color.opponent()) {
            self.discoverable_checks |= ray_containing(self.king_square, checker) ^ checker;
        }

        // The "checkmask" determines what squares we can legally move *to*
        // If there are no checkers, the checkmask contains any square that is occupied by the enemy or is empty.
        if self.checkers.is_empty() {
            self.checkmask = self.enemy_or_empty(color);
        }
        // Otherwise, the checkmask is the path from the checker(s) to the King, because we must either block the check or capture the checker.
        else {
            self.checkmask = Bitboard::EMPTY_BOARD;

            // There is *usually* only one checker, so this rarely loops.
            for checker_square in self.checkers {
                self.checkmask |= ray_between(self.king_square, checker_square);
            }

            // OR with the checkers to include them in the checkmask (since there is no ray between a King and a Knight)
            self.checkmask |= self.checkers
        };
    }
}

impl Deref for Game {
    type Target = Position;
    fn deref(&self) -> &Self::Target {
        &self.position
    }
}

impl FromStr for Game {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_fen(s)
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new(Position::default())
    }
}

impl<'a> IntoIterator for &'a Game {
    type IntoIter = MoveGenIter<'a>;
    type Item = Move;
    fn into_iter(self) -> Self::IntoIter {
        MoveGenIter::new(self)
    }
}

impl<'a> IntoIterator for &'a mut Game {
    type IntoIter = MoveGenIter<'a>;
    type Item = Move;
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
            (self.pinmask, "Pinmask"),
            (self.discoverable_checks, "Disc. Checks"),
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
