use std::fmt;

use super::{
    Board, CastlingRights, Color, Piece, PieceKind, Position, Rank, Square, XoShiRo,
    NUM_CASTLING_RIGHTS,
};

/// Stores Zobrist hash keys, for hashing [`Position`]s.
///
/// Initialized upon program startup with library-supplied keys that remain constant between compilations.
const ZOBRIST_TABLE: ZobristHashTable = ZobristHashTable::new();

/// Represents a key generated from a Zobrist Hash
#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
pub struct ZobristKey(u64);

impl ZobristKey {
    /// Generates a new [`ZobristKey`] from the supplied [`Position`].
    pub fn new(position: &Position) -> Self {
        Self::from_parts(
            position.board(),
            position.ep_square(),
            position.castling_rights(),
            position.side_to_move(),
        )
    }

    /// Generates a [`ZobristKey`] from the provided components of a [`Position`].
    pub fn from_parts(
        board: &Board,
        ep_square: Option<Square>,
        castling_rights: &CastlingRights,
        color: Color,
    ) -> Self {
        let mut key = Self::default();

        // Hash all pieces on the board
        for (square, piece) in board.iter() {
            key.hash_piece(square, piece);
        }

        // Hash the en passant square, if it exists
        key.hash_optional_ep_square(ep_square);

        // Hash the castling rights
        key.hash_castling_rights(castling_rights);

        // Hash the side-to-move
        key.hash_side_to_move(color);

        key
    }

    /// Return the inner `u64` of this key.
    pub fn inner(&self) -> u64 {
        self.0
    }

    /// Adds/removes `hash_key` to this [`ZobristKey`].
    ///
    /// This is done internally with the XOR operator.
    pub fn hash(&mut self, hash_key: u64) {
        self.0 ^= hash_key;
    }

    /// Adds/removes the hash for the provided `piece`/`square` combo to this [`ZobristKey`].
    pub fn hash_piece(&mut self, square: Square, piece: Piece) {
        self.hash(ZOBRIST_TABLE.piece_keys[square][piece]);
    }

    /// Adds/removes the hash for the provided `ep_square` to this [`ZobristKey`].
    pub fn hash_ep_square(&mut self, ep_square: Square) {
        self.hash(ZOBRIST_TABLE.ep_keys[ep_square]);
    }

    /// Adds/removes the hash for the provided `ep_square` to this [`ZobristKey`].
    pub fn hash_optional_ep_square(&mut self, ep_square: Option<Square>) {
        // This works because all squares where EP isn't possible (including Square::default) have a hash value of 0
        self.hash_ep_square(ep_square.unwrap_or_default());
    }

    /// Adds/removes the hash for the provided `castling_rights` to this [`ZobristKey`].
    pub fn hash_castling_rights(&mut self, castling_rights: &CastlingRights) {
        self.hash(ZOBRIST_TABLE.castling_keys[castling_rights]);
    }

    /// Adds/removes the hash for when the side-to-move is Black to this [`ZobristKey`].
    pub fn hash_side_to_move(&mut self, color: Color) {
        self.hash(ZOBRIST_TABLE.color_key[color]);
    }
}

impl fmt::Display for ZobristKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Encapsulates the logic of Zobrist hashing.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct ZobristHashTable {
    /// One unique key for every possible piece and every possible square.
    piece_keys: [[u64; PieceKind::COUNT]; Square::COUNT],

    /// One unique key for every square where en passant is possible.
    ep_keys: [u64; Square::COUNT],

    /// One key for every possible combination of castling rights.
    castling_keys: [u64; NUM_CASTLING_RIGHTS],

    /// One key for the side-to-move (only if side-to-move is Black- White's key is 0).
    color_key: [u64; Color::COUNT],
}

impl ZobristHashTable {
    /// Initialize this table, generating keys via the [`XoShiRo`] struct.
    ///
    /// This is only done once, at compilation, and is stored in the global `ZOBRIST_TABLE` constant.
    const fn new() -> Self {
        let mut piece_keys = [[0; PieceKind::COUNT]; Square::COUNT];
        let mut color_key = [0; Color::COUNT];
        let mut ep_keys = [0; Square::COUNT];
        let mut castling_keys = [0; NUM_CASTLING_RIGHTS];

        let mut prng = XoShiRo::new();

        // Initialize keys for pieces and EP
        let mut i = 0;
        while i < Square::COUNT {
            let mut j = 0;
            // Initialize keys for pieces
            while j < PieceKind::COUNT {
                let key;
                (key, prng) = prng.get_next_const();
                piece_keys[i][j] = key;
                j += 1;
            }

            // Initialize keys for en passant squares
            let rank = Square::from_index_unchecked(i).rank();
            if rank.is(&Rank::THREE) || rank.is(&Rank::SIX) {
                // Since en passant can only happen on ranks 3 and 6, we only need to store hash keys for those ranks
                let key;
                (key, prng) = prng.get_next_const();
                ep_keys[i] = key;
            }

            i += 1;
        }

        // Initialize keys for castling rights
        i = 0;
        while i < NUM_CASTLING_RIGHTS {
            let key;
            (key, prng) = prng.get_next_const();
            castling_keys[i] = key;
            i += 1;
        }

        // Initialize keys for side-to-move
        // Only Black has a key, since White's is just 0
        let (key, _) = prng.get_next_const();
        color_key[Color::Black.index()] = key;

        Self {
            piece_keys,
            ep_keys,
            castling_keys,
            color_key,
        }
    }
}
