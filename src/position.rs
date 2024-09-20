use std::{
    fmt::{self, Write},
    ops::{Deref, Index, IndexMut},
    str::FromStr,
};

use anyhow::{anyhow, bail, Result};

use super::{
    Bitboard, Color, File, Move, MoveKind, Piece, PieceKind, Rank, Square, ZobristKey,
    FEN_STARTPOS, NUM_CASTLING_RIGHTS,
};

// TODO: Refactor this to be Option<square> instead of bool arrays
/// Represents the castling rights of both players
#[derive(Clone, PartialEq, Eq, Debug, Hash, Default)]
pub struct CastlingRights {
    /// If a right is `Some(square)`, then `square` is the *rook*'s location
    pub(crate) short: [Option<Square>; Color::COUNT],
    pub(crate) long: [Option<Square>; Color::COUNT],
}

impl CastlingRights {
    pub const fn new() -> Self {
        Self {
            short: [None; Color::COUNT],
            long: [None; Color::COUNT],
        }
    }

    pub fn from_uci(uci: &str) -> Result<Self> {
        let mut short = [None; Color::COUNT];
        let mut long = [None; Color::COUNT];

        if uci.contains(['K', 'k', 'Q', 'q']) {
            short[Color::White] = uci.contains('K').then_some(Square::H1);
            long[Color::White] = uci.contains('Q').then_some(Square::A1);
            short[Color::Black] = uci.contains('k').then_some(Square::H8);
            long[Color::Black] = uci.contains('q').then_some(Square::A8);
        } else if uci.chars().any(|c| File::from_char(c).is_ok()) {
            eprintln!("Warning: Chess960 FEN detected for castling rights: {uci:?}");
            eprintln!("Chess960 is not currently supported");
            /*
            // TODO: Support Chess960
            for c in uci.chars() {
                let color = Color::from_bool(c.is_ascii_lowercase());
                let file = File::from_char(c)?;
                let rank = Rank::first(color);
                let rook_square = Square::new(file, rank);

                let king_file = File::E; // TODO: Fetch King's file the rest of the FEN
                if file > king_file {
                    short[color] = Some(rook_square);
                } else {
                    long[color] = Some(rook_square);
                }
            }
             */
        }

        Ok(Self { short, long })
    }

    pub fn to_uci(&self) -> String {
        let mut castling = String::with_capacity(4);

        if self.short[Color::White].is_some() {
            castling.push('K');
        }
        if self.long[Color::White].is_some() {
            castling.push('Q');
        }
        if self.short[Color::Black].is_some() {
            castling.push('k');
        }
        if self.long[Color::Black].is_some() {
            castling.push('q')
        }

        if castling.is_empty() {
            String::from("-")
        } else {
            castling
        }
    }

    /// Creates a `usize` for indexing into lists of 16 elements.
    ///
    /// # Example
    /// ```
    /// # use chessie::CastlingRights;
    /// let all = CastlingRights::from_uci("KQkq").unwrap();
    /// assert_eq!(all.index(), 15);
    /// let none = CastlingRights::from_uci("").unwrap();
    /// assert_eq!(none.index(), 0);
    /// ```
    pub const fn index(&self) -> usize {
        (self.short[0].is_some() as usize)
            | (self.short[1].is_some() as usize) << 1
            | (self.long[0].is_some() as usize) << 2
            | (self.long[1].is_some() as usize) << 3
    }
}

impl FromStr for CastlingRights {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_uci(s)
    }
}

impl<T> Index<CastlingRights> for [T; NUM_CASTLING_RIGHTS] {
    type Output = T;
    /// [`CastlingRights`] can be used to index into a list of 16 elements.
    fn index(&self, index: CastlingRights) -> &Self::Output {
        &self[index.index()]
    }
}

impl<'a, T> Index<&'a CastlingRights> for [T; NUM_CASTLING_RIGHTS] {
    type Output = T;
    /// [`CastlingRights`] can be used to index into a list of 16 elements.
    fn index(&self, index: &'a CastlingRights) -> &Self::Output {
        &self[index.index()]
    }
}

impl<T> IndexMut<CastlingRights> for [T; NUM_CASTLING_RIGHTS] {
    /// [`CastlingRights`] can be used to index into a list of 16 elements.
    fn index_mut(&mut self, index: CastlingRights) -> &mut Self::Output {
        &mut self[index.index()]
    }
}

impl<'a, T> IndexMut<&'a CastlingRights> for [T; NUM_CASTLING_RIGHTS] {
    /// [`CastlingRights`] can be used to index into a list of 16 elements.
    fn index_mut(&mut self, index: &'a CastlingRights) -> &mut Self::Output {
        &mut self[index.index()]
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// Represents the current state of the game, including move counters.
///
/// Analogous to a FEN string.
#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    /// Bitboard representation of the game board.
    board: Board,

    /// The [`Color`] of the current player.
    side_to_move: Color,

    /// Castling rights for each player.
    castling_rights: CastlingRights,

    /// Optional attack square for en passant.
    ep_square: Option<Square>,

    /// Used to enforce the fifty-move rule.
    /// - Incremented after each move.
    /// - Reset after a capture or a pawn moves.
    halfmove: usize,

    /// Number of moves since the beginning of the game.
    /// A fullmove is a complete turn by white and then by black.
    fullmove: usize,

    /// Zobrist hash key of this position
    key: ZobristKey,
}

impl Position {
    /// Creates a new, empty [`Position`] with the following properties:
    /// * No pieces on the board
    /// * White moves first
    /// * No castling rights
    /// * No en passant square available
    /// * Halfmove counter set to 0
    /// * Fullmove counter set to 1
    ///
    /// # Example
    /// ```
    /// # use chessie::Position;
    /// let state = Position::new();
    /// assert_eq!(state.to_fen(), "8/8/8/8/8/8/8/8 w - - 0 1");
    /// ```
    pub fn new() -> Self {
        let board = Board::new();
        let castling_rights = CastlingRights::new();
        let current_player = Color::White;
        let ep_square = None;

        let key = ZobristKey::from_parts(&board, ep_square, &castling_rights, current_player);

        Self {
            board,
            side_to_move: current_player,
            castling_rights,
            ep_square,
            halfmove: 0,
            fullmove: 1,
            key,
        }
    }

    /// Creates a new [`Position`] from the provided FEN string.
    pub fn from_fen(fen: &str) -> Result<Self> {
        let mut pos = Self::new();
        let mut split = fen.trim().split(' ');
        let placements = split.next().ok_or(anyhow!(
            "Invalid FEN string: FEN string must have piece placements."
        ))?;
        pos.board = Board::from_fen(placements)?;

        let active_color = split.next().unwrap_or("w");
        pos.side_to_move = Color::from_str(active_color)?;

        let castling = split.next().unwrap_or("KQkq");
        pos.castling_rights = CastlingRights::from_uci(castling)?;

        let en_passant_target = split.next().unwrap_or("-");
        pos.ep_square = match en_passant_target {
            "-" => None,
            square => Some(Square::from_uci(square)?),
        };

        let halfmove = split.next().unwrap_or("0");
        pos.halfmove = halfmove.parse().or(Err(anyhow!(
            "Invalid FEN string: FEN string must have valid halfmove counter. Got {halfmove}"
        )))?;

        let fullmove = split.next().unwrap_or("1");
        pos.fullmove = fullmove.parse().or(Err(anyhow!(
            "Invalid FEN string: FEN string must have valid fullmove counter. Got {fullmove}"
        )))?;

        pos.key = ZobristKey::new(&pos);

        Ok(pos)
    }

    /// Consumes `self` and returns a [`Position`] after having applied the provided [`Move`].
    pub fn with_move_made(mut self, mv: Move) -> Self {
        self.make_move(mv);
        self
    }

    /// Generates a FEN string from this [`Position`].
    pub fn to_fen(&self) -> String {
        let placements = self.board().to_fen();
        let active_color = self.side_to_move();
        let castling = self.castling_rights.to_uci();

        let en_passant_target = if let Some(square) = self.ep_square {
            square.to_string()
        } else {
            String::from("-")
        };

        let halfmove = self.halfmove;
        let fullmove = self.fullmove;

        format!("{placements} {active_color} {castling} {en_passant_target} {halfmove} {fullmove}")
    }

    /// Returns the current player as a [`Color`].
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// If en passant can be performed, returns the en passant [`Square`].
    pub const fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    /// If en passant can be performed, returns the destination of a pawn that would perform en passant.
    pub fn ep_target_square(&self) -> Option<Square> {
        self.ep_square()
            .map(|ep_square| ep_square.backward_by(self.side_to_move(), 1).unwrap())
    }

    /// Returns the [`CastlingRights`] of the current position.
    pub const fn castling_rights(&self) -> &CastlingRights {
        &self.castling_rights
    }

    /// Returns the half-move counter of the current position.
    pub const fn halfmove(&self) -> usize {
        self.halfmove
    }

    /// Returns the full-move counter of the current position.
    pub const fn fullmove(&self) -> usize {
        self.fullmove
    }

    /// Fetch the Zobrist hash key of this position.
    pub fn key(&self) -> ZobristKey {
        self.key
    }

    /// Returns `true` if the half-move counter is 100 or greater.
    ///
    /// Since "half-move" increases with ply, the 50-move rule takes effect at 100 ply.
    pub const fn can_draw_by_fifty(&self) -> bool {
        self.halfmove() >= 100
    }

    /// Toggles the current player from White to Black (or vice versa).
    pub fn toggle_current_player(&mut self) {
        self.side_to_move = self.side_to_move.opponent();
    }

    /// Fetches this position's [`Board`]
    pub const fn board(&self) -> &Board {
        &self.board
    }

    /// Mutably fetches this position's [`Board`]
    pub fn board_mut(&mut self) -> &mut Board {
        &mut self.board
    }

    /// Returns `true` if `color` can castle (either short or long).
    pub const fn can_castle(&self, color: Color) -> bool {
        self.castling_rights().short[color.index()].is_some()
            || self.castling_rights().long[color.index()].is_some()
    }

    /// Two positions are considered repetitions if they share the same piece layout, castling rights, and en passant square.
    ///
    /// Fullmove and Halfmove clocks are ignored.
    pub fn is_repetition_of(&self, other: &Self) -> bool {
        self.board() == other.board()
            && self.side_to_move() == other.side_to_move()
            && self.castling_rights() == other.castling_rights()
            && self.ep_square() == other.ep_square()
    }

    /// Checks if the provided move is legal to perform.
    ///
    /// If `Ok()`, the move is legal.
    /// If `Err(msg)`, then `msg` will be a reason as to why it's not legal.
    pub fn check_legality_of(&self, mv: Move) -> Result<()> {
        let (from, to, kind) = mv.parts();

        // If there's no piece here, illegal move
        let Some(piece) = self.board().piece_at(from) else {
            bail!("No piece here to move");
        };

        // If it's not this piece's color's turn, illegal move
        if piece.color() != self.side_to_move() {
            bail!("Tried to move a piece that wasn't yours");
        }

        // If this move captures a piece, handle those cases
        if let Some(to_capture) = self.board().piece_at(to) {
            // Can't capture own pieces
            if to_capture.color() == piece.color() {
                bail!("Tried to capture your own piece");
            }

            // Can't capture king
            if to_capture.is_king() {
                bail!("Tried to capture enemy king");
            }

            // Ensure that the move is a capture or en passant, and that it captures the correct piece
            if !mv.is_capture() {
                bail!("Captured on a non-capture move");
            }
        }

        match kind {
            // If the move is pawn-specific, ensure it's a pawn moving
            MoveKind::EnPassantCapture | MoveKind::PawnDoublePush => {
                if !piece.is_pawn() {
                    bail!("Tried to do a pawn move (EP, Push 2, Promote) with a piece that isn't a pawn");
                }
            }
            // If castling, ensure we have the right to
            MoveKind::ShortCastle => {
                if self.castling_rights.short[piece.color()].is_none() {
                    bail!("Tried to castle (short) without rights");
                }
            }
            // If castling, ensure we have the right to
            MoveKind::LongCastle => {
                if self.castling_rights.long[piece.color()].is_none() {
                    bail!("Tried to castle (long) without rights");
                }
            }
            // Quiet moves are fine
            _ => {}
        }

        Ok(())
    }

    /// Applies the move, if it is legal to make. If it is not legal, returns an `Err` explaining why.
    pub fn make_move_checked(&mut self, mv: Move) -> Result<()> {
        self.check_legality_of(mv)?;
        self.make_move(mv);
        Ok(())
    }

    /// Apply the provided `moves` to the board. No enforcement of legality.
    pub fn make_moves(&mut self, moves: impl IntoIterator<Item = Move>) {
        for mv in moves {
            self.make_move(mv);
        }
    }

    /// Applies the move. No enforcement of legality
    pub fn make_move(&mut self, mv: Move) {
        // Remove the piece from it's previous location, exiting early if there is no piece there
        let Some(mut piece) = self.board_mut().take(mv.from()) else {
            return;
        };

        let color = piece.color();
        let to = mv.to();
        let from = mv.from();

        // Un-hash the piece at `from`.
        self.key.hash_piece(from, piece);

        // Clear the EP square from the last move (and un-hash it)
        self.key.hash_optional_ep_square(self.ep_square.take());

        // Un-hash the castling rights
        self.key.hash_castling_rights(&self.castling_rights);

        // Increment move counters
        self.halfmove += 1; // This is reset if a capture occurs or a pawn moves
        self.fullmove += self.side_to_move().index();

        // First, deal with special cases like captures and castling
        if mv.is_capture() {
            // If this move was en passant, the piece we captured isn't at `to`, it's one square behind
            let captured_square = if mv.is_en_passant() {
                to.backward_by(color, 1).unwrap()
            } else {
                to
            };

            let Some(captured) = self.board_mut().take(captured_square) else {
                panic!("Failed to apply {mv:?} to {self}: No piece found at {captured_square}");
            };
            let captured_color = captured.color();

            // If the capture was on a rook's starting square, disable that side's castling.
            // Either a rook was captured, or there wasn't a rook there, in which case castling on that side is already disabled
            // TODO: Chess960
            if to == Square::A1.rank_relative_to(captured_color) {
                self.key.hash_castling_rights(&self.castling_rights);
                self.castling_rights.long[captured_color].take();
                self.key.hash_castling_rights(&self.castling_rights);
            }

            if to == Square::H1.rank_relative_to(captured_color) {
                self.key.hash_castling_rights(&self.castling_rights);
                self.castling_rights.short[captured_color].take();
                self.key.hash_castling_rights(&self.castling_rights);
            }

            // Reset halfmove counter, since a capture occurred
            self.halfmove = 0;
        } else if mv.is_pawn_double_push() {
            // Double pawn push, so set the EP square
            self.ep_square = from.forward_by(color, 1);
            self.key.hash_optional_ep_square(self.ep_square());
        } else if mv.is_castle() {
            let castle_index = mv.is_short_castle() as usize;
            // TODO: Chess960
            let old_rook_square = [Square::A1, Square::H1][castle_index].rank_relative_to(color);
            let new_rook_square = [Square::D1, Square::F1][castle_index].rank_relative_to(color);

            // Move the rook. The King is already handled before and after this match statement.
            let rook = self.board_mut().take(old_rook_square).unwrap();
            self.board_mut().place(rook, new_rook_square);

            // Disable castling
            // Hashing must be done before and after castling rights are changed so that the proper hash key is used
            self.key.hash_castling_rights(&self.castling_rights);
            self.castling_rights.short[color] = None;
            self.castling_rights.long[color] = None;
            self.key.hash_castling_rights(&self.castling_rights);
        }

        // Next, handle special cases for Pawn (halfmove), Rook, and King (castling)
        match piece.kind() {
            PieceKind::Pawn => self.halfmove = 0,
            PieceKind::Rook => {
                // Disable castling if a rook moved
                if from == Square::A1.rank_relative_to(color) {
                    self.key.hash_castling_rights(&self.castling_rights);
                    self.castling_rights.long[color].take();
                    self.key.hash_castling_rights(&self.castling_rights);
                }

                if from == Square::H1.rank_relative_to(color) {
                    self.key.hash_castling_rights(&self.castling_rights);
                    self.castling_rights.short[color].take();
                    self.key.hash_castling_rights(&self.castling_rights);
                }
            }
            PieceKind::King => {
                // Disable all castling
                self.key.hash_castling_rights(&self.castling_rights);
                self.castling_rights.short[color] = None;
                self.castling_rights.long[color] = None;
                self.key.hash_castling_rights(&self.castling_rights);
            }
            _ => {}
        }

        // Now we check for promotions, since all special cases for Pawns and Rooks have been dealt with
        if let Some(promotion) = mv.promotion() {
            piece = piece.promoted(promotion);
        }

        // Place the piece in it's new position
        self.board_mut().place(piece, to);

        // Hash the piece at `to`.
        self.key.hash_piece(to, piece);

        // Next player's turn
        self.toggle_current_player();

        // Toggle the hash of the current player
        self.key.hash_side_to_move(self.side_to_move());
    }
}

impl FromStr for Position {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_fen(s)
    }
}

impl Deref for Position {
    type Target = Board;
    fn deref(&self) -> &Self::Target {
        self.board()
    }
}

impl Default for Position {
    fn default() -> Self {
        // Safe unwrap because the FEN for startpos is always valid
        Self::from_fen(FEN_STARTPOS).unwrap()
    }
}

impl fmt::Display for Position {
    /// Display this position's FEN string
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fen())
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ranks = Rank::iter().rev();

        let mut board_str = String::with_capacity(10);
        for rank in ranks {
            board_str += &format!("{rank}|");
            for file in File::iter() {
                let piece = self.board().piece_at(file * rank);
                let piece_char = piece.map(|p| p.char()).unwrap_or('.');
                board_str += &format!(" {piece_char}");
            }

            if rank == Rank::SEVEN {
                board_str += &format!("           FEN: {}", self.to_fen());
            } else if rank == Rank::SIX {
                board_str += &format!("          Side: {}", self.side_to_move());
            } else if rank == Rank::FIVE {
                board_str += &format!("      Castling: {}", self.castling_rights());
            } else if rank == Rank::FOUR {
                let ep = self
                    .ep_square()
                    .map(|t| t.to_uci())
                    .unwrap_or(String::from("-"));
                board_str += &format!("            EP: {ep}",);
            } else if rank == Rank::THREE {
                board_str += &format!("     Half-move: {}", self.halfmove());
            } else if rank == Rank::TWO {
                board_str += &format!("     Full-move: {}", self.fullmove());
            }
            board_str += "\n";
        }
        board_str += " +";
        for _ in File::iter() {
            board_str += "--";
        }
        board_str += "\n   ";
        for file in File::iter() {
            board_str += &format!("{file} ");
        }

        write!(f, "{board_str}")
    }
}

/// Represents all pieces and their locations on a chess board.
///
/// Has no knowledge of castling rights, en passant, or move counters. If you need those, see [`Position`].
///
/// Internally uses a collection of [`Bitboard`]s to keep track of piece/color locations.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Board {
    /// All squares occupied by a specific color.
    colors: [Bitboard; Color::COUNT],

    /// All squares occupied by a specific piece kind.
    pieces: [Bitboard; PieceKind::COUNT],

    /// Redundant mailbox to speed up the [`Board::piece_at`] functions.
    mailbox: [Option<Piece>; Square::COUNT],
}

impl Board {
    /// Creates a new, empty [`Board`] containing no pieces.
    ///
    /// # Example
    /// ```
    /// # use chessie::Board;
    /// let board = Board::new();
    /// assert_eq!(board.to_fen(), "8/8/8/8/8/8/8/8");
    /// ```
    pub const fn new() -> Self {
        Self {
            colors: [Bitboard::EMPTY_BOARD; Color::COUNT],
            pieces: [Bitboard::EMPTY_BOARD; PieceKind::COUNT],
            mailbox: [None; Square::COUNT],
        }
    }

    /// Constructs a [`Board`] from the provided FEN string, ignoring castling/ep/move counters.
    pub fn from_fen(fen: &str) -> Result<Self> {
        let mut board = Self::new();

        // If this FEN string contains more than just the initial placements, extract the placements
        let placements = if fen.contains(' ') {
            fen.split(' ').next().unwrap()
        } else {
            fen
        };

        // Check if the placements string is the correct length
        if placements.matches('/').count() != 7 {
            bail!("Missing placements for all 8 ranks.");
        }

        // Need to reverse this so that White pieces are at the "bottom" of the board
        for (rank, placements) in placements.split('/').rev().enumerate() {
            let mut file = 0;
            let rank = rank as u8;

            for piece_char in placements.chars() {
                // If the next char is a piece, we need to update the relevant Bitboards
                if let Ok(piece) = Piece::from_uci(piece_char) {
                    // Firstly, create a square and set the "Occupied" board at this location.
                    let square = Square::new(File::new_unchecked(file), Rank::new_unchecked(rank));

                    board.place(piece, square);

                    file += 1;
                } else {
                    // If the next char was not a piece, increment our File counter, checking for errors along the way
                    let Some(empty) = piece_char.to_digit(10) else {
                        bail!("Found non-piece, non-numeric char {piece_char:?} when parsing FEN.");
                    };
                    file += empty as u8
                }
            }
        }

        Ok(board)
    }

    /*
    /// Returns an instance of this [`Board`] that has the additional bits specified by `mask` set, according to the [`Piece`] supplied.
    ///
    /// If `mask` contains only 1 square, use [`Board::with`] instead, as it is likely to be faster.
    pub const fn with(self, mask: Bitboard, piece: Piece) -> Self {
        let (color, kind) = piece.parts();

        let mut colors = self.colors;
        colors[color.index()] = colors[color.index()].or(mask);

        let mut pieces = self.pieces;
        pieces[kind.index()] = pieces[kind.index()].or(mask);

        Self { colors, pieces }
    }
     */

    /*
    /// Returns an instance of this [`Board`] that has all bits specified by `mask` cleared.
    pub fn without(self, mask: Bitboard) -> Self {
        let not_mask = !mask;

        let mut colors = self.colors;
        for color in Color::all() {
            colors[color] &= not_mask;
        }

        let mut pieces = self.pieces;
        for kind in PieceKind::all() {
            pieces[kind] &= not_mask;
        }

        Self { colors, pieces }
    }
      */

    /// Returns `true` if there is a piece at the given [`Square`], else `false`.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Square};
    /// let board = Board::default();
    /// assert_eq!(board.has(Square::B1), true);
    /// ```
    pub const fn has(&self, square: Square) -> bool {
        self.mailbox[square.index()].is_some()
    }

    /// Places the provided [`Piece`] and the supplied [`Square`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Piece, PieceKind, Color, Square};
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// let mut board = Board::new();
    /// board.place(white_knight, Square::C4);
    /// assert_eq!(board.to_fen(), "8/8/8/8/2N5/8/8/8");
    /// ```
    pub fn place(&mut self, piece: Piece, square: Square) {
        self[piece.color()].set(square);
        self[piece.kind()].set(square);
        self.mailbox[square] = Some(piece);
    }

    /// Clears the supplied [`Square`] of any pieces.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Square};
    /// let mut board = Board::from_fen("k7/8/8/8/2N5/8/8/7K").unwrap();
    /// board.clear(Square::C4);
    /// assert_eq!(board.to_fen(), "k7/8/8/8/8/8/8/7K");
    /// ```
    pub fn clear(&mut self, square: Square) {
        // if let Some(piece) = self.piece_at(square) {
        //     // if let Some(piece) = self.mailbox[square].take() {
        //     self[piece.color()].clear(square);
        //     self[piece.kind()].clear(square);
        //     self.mailbox[square] = None;
        // }

        self.take(square);
    }

    /// Takes the [`Piece`] from a given [`Square`], if there is one present.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Piece, PieceKind, Color, Square};
    /// let mut board = Board::from_fen("k7/8/8/8/2N5/8/8/7K").unwrap();
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// let taken = board.take(Square::C4);
    /// assert_eq!(board.to_fen(), "k7/8/8/8/8/8/8/7K");
    /// assert_eq!(taken, Some(white_knight));
    /// ```
    pub fn take(&mut self, square: Square) -> Option<Piece> {
        // Take the piece from the mailbox, exiting early if there is none
        let piece = self.mailbox[square].take()?;

        // If there was a piece, clear the internal bitboards.
        self.colors[piece.color()].clear(square);
        self.pieces[piece.kind()].clear(square);

        Some(piece)
    }

    /// Clears the entire board, removing all pieces.
    ///
    /// # Example
    /// ```
    /// # use chessie::Board;
    /// let mut board = Board::default();
    /// board.clear_all();
    /// assert_eq!(board.to_fen(), "8/8/8/8/8/8/8/8");
    /// ```
    pub fn clear_all(&mut self) {
        *self = Self::new();
    }

    /// Fetches the [`Color`] of the piece at the provided [`Square`], if there is one.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Color, Square};
    /// let board = Board::default();
    /// assert_eq!(board.color_at(Square::A2), Some(Color::White));
    /// assert_eq!(board.color_at(Square::E8), Some(Color::Black));
    /// assert!(board.color_at(Square::E4).is_none());
    /// ```
    pub fn color_at(&self, square: Square) -> Option<Color> {
        self.mailbox[square].map(|piece| piece.color())
    }

    /// Fetches the [`PieceKind`] of the piece at the provided [`Square`], if there is one.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, PieceKind, Square};
    /// let mut board = Board::default();
    /// assert_eq!(board.kind_at(Square::A2), Some(PieceKind::Pawn));
    /// assert!(board.kind_at(Square::E4).is_none());
    /// ```
    pub fn kind_at(&self, square: Square) -> Option<PieceKind> {
        self.mailbox[square].map(|piece| piece.kind())
    }

    /// Fetches the [`Piece`] of the piece at the provided [`Square`], if there is one.
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, PieceKind, Color, Square};
    /// let mut board = Board::default();
    /// assert_eq!(board.piece_at(Square::A2).unwrap().kind(), PieceKind::Pawn);
    /// assert_eq!(board.piece_at(Square::A2).unwrap().color(), Color::White);
    /// assert!(board.piece_at(Square::E4).is_none());
    /// ```
    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.mailbox[square.index()]
    }

    /// Fetches the [`Piece`] of the piece at the provided [`Square`], without checking if one is there.
    ///
    /// This is an internal function, and should never be called unless you know what you're doing (hint: you probably don't).
    pub(crate) fn piece_at_unchecked(&self, square: Square) -> Piece {
        unsafe { self.piece_at(square).unwrap_unchecked() }
    }

    /// Fetches the [`Bitboard`] corresponding to the supplied [`PieceKind`].
    ///
    /// The returned [`Bitboard`] will hold the locations of every occurrence of each [`Piece`] matching the supplied [`PieceKind`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, PieceKind, Bitboard};
    /// let board = Board::default();
    /// let pawns = board.kind(PieceKind::Pawn);
    /// assert_eq!(pawns, Bitboard::RANK_2 | Bitboard::RANK_7);
    /// ```
    pub const fn kind(&self, kind: PieceKind) -> Bitboard {
        self.pieces[kind.index()]
    }

    /// Fetches the [`Bitboard`] corresponding to the supplied [`Color`].
    ///
    /// The returned [`Bitboard`] will hold the locations of every occurrence each [`Piece`] matching the supplied [`Color`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, Color, Piece, Bitboard};
    /// let board = Board::default();
    /// let white_pieces = board.color(Color::White);
    /// assert_eq!(white_pieces, Bitboard::RANK_1 | Bitboard::RANK_2);
    /// ```
    pub const fn color(&self, color: Color) -> Bitboard {
        self.colors[color.index()]
    }

    /// Fetches a [`Bitboard`] of all occupied squares on the board.
    pub const fn occupied(&self) -> Bitboard {
        self.color(Color::White).or(self.color(Color::Black))
    }

    /// Fetches a [`Bitboard`] of all non-occupied squares on the board.
    pub const fn empty(&self) -> Bitboard {
        self.occupied().not()
    }

    /// Fetches the [`Bitboard`] corresponding to the supplied [`Piece`].
    ///
    /// The returned [`Bitboard`] will hold the locations of every occurrence of the supplied [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use chessie::{Board, PieceKind, Color, Piece, Bitboard};
    /// let board = Board::default();
    /// let white_pawn = Piece::new(Color::White, PieceKind::Pawn);
    /// let white_pawns = board.piece(white_pawn);
    /// assert_eq!(white_pawns, Bitboard::RANK_2);
    /// ```
    pub const fn piece(&self, piece: Piece) -> Bitboard {
        self.piece_parts(piece.color(), piece.kind())
    }

    /// Creates a [`BoardIter`] to iterate over all occupied [`Square`]s in this [`Board`].
    pub const fn iter(&self) -> BoardIter<'_> {
        BoardIter {
            board: self,
            occupancy: self.occupied(),
        }
    }

    /// Returns an iterator over all of the pieces of `kind` on this board along with their corresponding locations.
    pub const fn all_of(&self, kind: PieceKind) -> BoardIter<'_> {
        BoardIter {
            board: self,
            occupancy: self.kind(kind),
        }
    }

    /// Returns an iterator over all of the pieces of `color` on this board along with their corresponding locations.
    pub const fn all_for(&self, color: Color) -> BoardIter<'_> {
        BoardIter {
            board: self,
            occupancy: self.color(color),
        }
    }

    /// Analogous to [`Board::piece`] with a [`Piece`]'s individual components.
    ///
    /// If you have a [`PieceKind`] and a [`Color`] already, this is likely to be *slightly*
    /// faster that constructing a [`Piece`] and calling [`Board::piece`].
    pub const fn piece_parts(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.color(color).and(self.kind(kind))
    }

    /// Fetches a [`Bitboard`] containing the locations of all orthogonal sliding pieces (Rook, Queen).
    pub fn orthogonal_sliders(&self, color: Color) -> Bitboard {
        (self.kind(PieceKind::Rook) | self.kind(PieceKind::Queen)) & self.color(color)
    }

    /// Fetches a [`Bitboard`] containing the locations of all diagonal sliding pieces (Bishop, Queen).
    pub fn diagonal_sliders(&self, color: Color) -> Bitboard {
        (self.kind(PieceKind::Bishop) | self.kind(PieceKind::Queen)) & self.color(color)
    }

    /// Fetches a [`Bitboard`] containing the locations of all sliding pieces (Rook, Bishop, Queen).
    pub fn sliders(&self, color: Color) -> Bitboard {
        (self.kind(PieceKind::Rook) | self.kind(PieceKind::Bishop) | self.kind(PieceKind::Queen))
            & self.color(color)
    }

    /// Fetches the [`Bitboard`] for the King of the provided color.
    pub const fn king(&self, color: Color) -> Bitboard {
        self.piece_parts(color, PieceKind::King)
    }

    /// Fetches the [`Bitboard`] for the Pawns of the provided color.
    pub const fn pawns(&self, color: Color) -> Bitboard {
        self.piece_parts(color, PieceKind::Pawn)
    }

    /// Fetches the [`Bitboard`] for the Knights of the provided color.
    pub const fn knights(&self, color: Color) -> Bitboard {
        self.piece_parts(color, PieceKind::Knight)
    }

    /// Get all squares that are either empty or occupied by the enemy
    ///
    /// # Example
    /// ```
    /// # use chessie::{Bitboard, Board, Color};
    /// let board = Board::default();
    /// let not_white = board.enemy_or_empty(Color::White);
    /// assert_eq!(not_white.to_hex_string(), "0xFFFFFFFFFFFF0000");
    /// ```
    pub const fn enemy_or_empty(&self, color: Color) -> Bitboard {
        self.color(color).not()
    }

    /// Generates a [FEN](https://www.chess.com/terms/fen-chess) string of this [`Board`].
    pub fn to_fen(&self) -> String {
        let mut placements: [String; 8] = Default::default();

        for rank in Rank::iter() {
            let mut empty_spaces = 0;
            for file in File::iter() {
                if let Some(piece) = self.piece_at(file * rank) {
                    if empty_spaces != 0 {
                        placements[rank.index()] += &empty_spaces.to_string();
                        empty_spaces = 0;
                    }
                    placements[rank.index()] += piece.as_ref();
                } else {
                    empty_spaces += 1;
                }
            }

            if empty_spaces != 0 {
                placements[rank.index()] += &empty_spaces.to_string();
            }
        }
        placements.reverse();

        placements.join("/")
    }
}

impl Default for Board {
    fn default() -> Self {
        // Safe unwrap because the FEN for startpos is always valid
        Self::from_fen(FEN_STARTPOS).unwrap()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Allocate just enough capacity
        let mut board = String::with_capacity(198);

        for rank in Rank::iter().rev() {
            board += &format!("{rank}| ");

            for file in File::iter() {
                let square = Square::new(file, rank);
                let occupant = if let Some(piece) = self.piece_at(square) {
                    piece.to_string()
                } else {
                    // String::from(if square.is_light() { "#" } else { "-" })
                    String::from(".")
                };

                board += &format!("{occupant} ");
            }

            board += "\n"
        }
        board += " +";
        for _ in File::iter() {
            board += "--";
        }
        board += "\n   ";
        for file in File::iter() {
            board += &format!("{file} ");
        }

        write!(f, "{board}")
    }
}

impl From<[Option<Piece>; 64]> for Board {
    fn from(value: [Option<Piece>; 64]) -> Self {
        let mut board = Self::new();

        for (i, piece) in value.into_iter().enumerate() {
            if let Some(piece) = piece {
                board.place(piece, Square::from_index(i).unwrap())
            }
        }

        board
    }
}

impl Index<PieceKind> for Board {
    type Output = Bitboard;
    fn index(&self, index: PieceKind) -> &Self::Output {
        &self.pieces[index]
    }
}

impl IndexMut<PieceKind> for Board {
    fn index_mut(&mut self, index: PieceKind) -> &mut Self::Output {
        &mut self.pieces[index]
    }
}

impl Index<Color> for Board {
    type Output = Bitboard;
    fn index(&self, index: Color) -> &Self::Output {
        &self.colors[index]
    }
}

impl IndexMut<Color> for Board {
    fn index_mut(&mut self, index: Color) -> &mut Self::Output {
        &mut self.colors[index]
    }
}

impl Index<Square> for Board {
    type Output = Option<Piece>;
    fn index(&self, index: Square) -> &Self::Output {
        &self.mailbox[index]
    }
}

impl IndexMut<Square> for Board {
    fn index_mut(&mut self, index: Square) -> &mut Self::Output {
        &mut self.mailbox[index]
    }
}

impl<'a> IntoIterator for &'a Board {
    type IntoIter = BoardIter<'a>;
    type Item = <BoardIter<'a> as Iterator>::Item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Board {
    type IntoIter = BoardIter<'a>;
    type Item = <BoardIter<'a> as Iterator>::Item;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl fmt::Debug for Board {
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

        let pieces = format(&[
            (self.pieces[PieceKind::Pawn], "Pawn"),
            (self.pieces[PieceKind::Knight], "Knight"),
            (self.pieces[PieceKind::Bishop], "Bishop"),
            (self.pieces[PieceKind::Rook], "Rook"),
            (self.pieces[PieceKind::Queen], "Queen"),
            (self.pieces[PieceKind::King], "King"),
        ]);

        let metadata = format(&[
            (self.occupied(), "Occupied"),
            (self.empty(), "Empty"),
            (self.colors[Color::White], "White"),
            (self.colors[Color::Black], "Black"),
        ]);

        write!(f, "Bitboards:\n{pieces}\n\n{metadata}")
    }
}

/// An iterator over a set of squares on a [`Board`].
///
/// Calls to [`Iterator::next`] will yield a tuple of a [`Square`] and a [`Piece`].
pub struct BoardIter<'a> {
    /// The board to retrieve pieces from.
    board: &'a Board,

    /// The list of squares to iterate over.
    occupancy: Bitboard,
}

impl<'a> Iterator for BoardIter<'a> {
    type Item = (Square, Piece);

    fn next(&mut self) -> Option<Self::Item> {
        let square = self.occupancy.pop_lsb()?;

        // Safety: Because we early return when calling `pop_lsb` above, there is guaranteed to be a piece at `square`.
        let piece = self.board.piece_at_unchecked(square);
        Some((square, piece))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.occupancy.population() as usize;
        (size, Some(size))
    }
}

impl<'a> ExactSizeIterator for BoardIter<'a> {}
