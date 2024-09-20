use std::{
    fmt,
    ops::{Index, Not, Shl, Shr},
    str::FromStr,
};

use anyhow::{anyhow, bail};

use super::{Color, File, Rank, Square};

/// A [`Bitboard`] represents the game board as a set of bits.
/// They are used for various computations, such as fetching valid moves or computing move costs.
///
/// The internal representation is a 64-bit binary number, so the values will represent the entire board.
/// They are color-agnostic, with the low order bits representing the "lower" half of the board.
///
/// Bit index 0 is the least-significant bit (LSB = 2^0)
/// Bit index 63 is the most-significant bit (MSB = 2^63)
///
/// The internal encoding uses [Little-Endian Rank-File Mapping (LERF)](https://www.chessprogramming.org/Square_Mapping_Considerations#Little-Endian_Rank-File_Mapping),
/// so a bitboard of first Rank would look like this in binary:
/// ```text
/// 00000000
/// 00000000
/// 00000000
/// 00000000
/// 00000000
/// 00000000
/// 00000000
/// 11111111
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(pub(crate) u64);

impl Bitboard {
    pub const FILE_A: Self = Self(0x0101010101010101);
    pub const FILE_B: Self = Self(0x0202020202020202);
    pub const FILE_C: Self = Self(0x0404040404040404);
    pub const FILE_D: Self = Self(0x0808080808080808);
    pub const FILE_E: Self = Self(0x1010101010101010);
    pub const FILE_F: Self = Self(0x2020202020202020);
    pub const FILE_G: Self = Self(0x4040404040404040);
    pub const FILE_H: Self = Self(0x8080808080808080);
    pub const NOT_FILE_A: Self = Self(0xfefefefefefefefe);
    pub const NOT_FILE_H: Self = Self(0x7f7f7f7f7f7f7f7f);
    pub const RANK_1: Self = Self(0x00000000000000FF);
    pub const RANK_2: Self = Self(0x000000000000FF00);
    pub const RANK_3: Self = Self(0x0000000000FF0000);
    pub const RANK_4: Self = Self(0x00000000FF000000);
    pub const RANK_5: Self = Self(0x000000FF00000000);
    pub const RANK_6: Self = Self(0x0000FF0000000000);
    pub const RANK_7: Self = Self(0x00FF000000000000);
    pub const RANK_8: Self = Self(0xFF00000000000000);
    pub const A1_H8_DIAG: Self = Self(0x8040201008040201);
    pub const H1_A8_DIAG: Self = Self(0x0102040810204080);
    pub const LIGHT_SQUARES: Self = Self(0x55AA55AA55AA55AA);
    pub const DARK_SQUARES: Self = Self(0xAA55AA55AA55AA55);
    pub const EMPTY_BOARD: Self = Self(0x0000000000000000);
    pub const FULL_BOARD: Self = Self(0xFFFFFFFFFFFFFFFF);
    pub const EDGES: Self = Self(0xFF818181818181FF);
    pub const CORNERS: Self = Self(0x8100000000000081);
    pub const CENTER: Self = Self(0x0000001818000000);
    pub const BACK_RANKS: Self = Self(0xFF000000000000FF);
    pub const PAWN_RANKS: Self = Self(0x00FF00000000FF00);

    /// Constructs a new [`Bitboard`] from the provided bit pattern.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let board = Bitboard::new(255);
    /// assert_eq!(board.to_hex_string(), "0x00000000000000FF");
    /// ```
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    /// Constructs a new [`Bitboard`] from the provided index.
    ///
    /// The resulting [`Bitboard`] will have only a single bit toggled on.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let board = Bitboard::from_index(63);
    /// assert_eq!(board.to_hex_string(), "0x8000000000000000");
    /// ```
    pub const fn from_index(index: usize) -> Self {
        debug_assert!(index < 64, "Index must be between [0,64)");
        Self::from_square(Square::from_index_unchecked(index))
    }

    /// Constructs a new [`Bitboard`] from the provided [`Square`].
    ///
    /// The resulting [`Bitboard`] will have only a single bit toggled on.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let board = Bitboard::from_square(Square::H8);
    /// assert_eq!(board.to_hex_string(), "0x8000000000000000");
    /// ```
    pub const fn from_square(square: Square) -> Self {
        Self(1 << square.index())
    }

    /// Constructs a new [`Bitboard`] from the provided [`File`].
    ///
    /// The resulting [`Bitboard`] will have an entire column of bits toggled on.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, File};
    /// let board = Bitboard::from_file(File::F);
    /// assert_eq!(board.to_hex_string(), "0x2020202020202020");
    /// ```
    pub const fn from_file(file: File) -> Self {
        Self::new(Self::FILE_A.0 << file.0)
    }

    /// Constructs a new [`Bitboard`] from the provided [`Rank`].
    ///
    /// The resulting [`Bitboard`] will have an entire row of bits toggled on.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Rank};
    /// let board = Bitboard::from_rank(Rank::SEVEN);
    /// assert_eq!(board.to_hex_string(), "0x00FF000000000000");
    /// ```
    pub const fn from_rank(rank: Rank) -> Self {
        Self::new(Self::RANK_1.0 << (rank.0 * 8))
    }

    /// Returns [`Bitboard::FULL_BOARD`] if `true`, else [`Bitboard::EMPTY_BOARD`].
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    ///
    /// assert_eq!(Bitboard::from_bool(true), Bitboard::FULL_BOARD);
    /// assert_eq!(Bitboard::from_bool(false), Bitboard::EMPTY_BOARD);
    /// ```
    pub const fn from_bool(value: bool) -> Self {
        Self((value as u64).wrapping_neg() & u64::MAX)
    }

    /// If `value` is `Some`, this converts the inner `T` using the appropriate [`Bitboard::from`] implementation.
    ///
    /// If `value` is `None`, this yields an empty bitboard.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// assert_eq!(Bitboard::from_option(Some(Square::A1)), Square::A1.bitboard());
    /// assert_eq!(Bitboard::from_option::<Square>(None), Bitboard::EMPTY_BOARD);
    /// ```
    pub fn from_option<T>(value: Option<T>) -> Self
    where
        Self: From<T>,
    {
        if let Some(t) = value {
            Self::from(t)
        } else {
            Self::default()
        }
    }

    /// Returns a [`Bitboard`] of this [`Color`]'s first rank.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    ///
    /// assert_eq!(Bitboard::first_rank(Color::White), Bitboard::RANK_1);
    /// assert_eq!(Bitboard::first_rank(Color::Black), Bitboard::RANK_8);
    /// ```
    pub const fn first_rank(color: Color) -> Self {
        [Self::RANK_1, Self::RANK_8][color.index()]
    }

    /// Returns a [`Bitboard`] of this [`Color`]'s second rank.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    ///
    /// assert_eq!(Bitboard::second_rank(Color::White), Bitboard::RANK_2);
    /// assert_eq!(Bitboard::second_rank(Color::Black), Bitboard::RANK_7);
    /// ```
    pub const fn second_rank(color: Color) -> Self {
        [Self::RANK_2, Self::RANK_7][color.index()]
    }

    /// Returns a [`Bitboard`] of this [`Color`]'s third rank.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    ///
    /// assert_eq!(Bitboard::third_rank(Color::White), Bitboard::RANK_3);
    /// assert_eq!(Bitboard::third_rank(Color::Black), Bitboard::RANK_6);
    /// ```
    pub const fn third_rank(color: Color) -> Self {
        [Self::RANK_3, Self::RANK_6][color.index()]
    }

    /// Returns a [`Bitboard`] of this [`Color`]'s seventh rank.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    ///
    /// assert_eq!(Bitboard::seventh_rank(Color::White), Bitboard::RANK_7);
    /// assert_eq!(Bitboard::seventh_rank(Color::Black), Bitboard::RANK_2);
    /// ```
    pub const fn seventh_rank(color: Color) -> Self {
        [Self::RANK_7, Self::RANK_2][color.index()]
    }

    /// Returns a [`Bitboard`] of this [`Color`]'s eighth (last) rank.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    ///
    /// assert_eq!(Bitboard::eighth_rank(Color::White), Bitboard::RANK_8);
    /// assert_eq!(Bitboard::eighth_rank(Color::Black), Bitboard::RANK_1);
    /// ```
    pub const fn eighth_rank(color: Color) -> Self {
        [Self::RANK_8, Self::RANK_1][color.index()]
    }

    /// Returns the inner `u64` of this [`Bitboard`].
    pub const fn inner(&self) -> u64 {
        self.0
    }

    /// Creates a [`Square`] from this [`Bitboard`] based on the lowest-index bit that is flipped.
    ///
    /// If this [`Bitboard`] contains more than a single flipped bit, it is converted into a [`Square`]
    /// based on the index of the lowest bit that is flipped.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let board = Bitboard::from_index(14);
    /// assert_eq!(board.to_square_unchecked(), Square::G2);
    /// ```
    pub const fn to_square_unchecked(&self) -> Square {
        Square::from_index_unchecked(self.0.trailing_zeros() as usize)
    }

    /// Creates a [`Square`] from this [`Bitboard`] based on the lowest-index bit that is flipped.
    ///
    /// If this [`Bitboard`] contains more than a single flipped bit, it is converted into a [`Square`]
    /// based on the index of the lowest bit that is flipped.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let board = Bitboard::from_index(14);
    /// assert_eq!(board.to_square(), Some(Square::G2));
    /// let invalid = Bitboard::RANK_1;
    /// assert_eq!(invalid.to_square(), None);
    /// ```
    pub const fn to_square(&self) -> Option<Square> {
        if self.population() == 1 {
            Some(self.to_square_unchecked())
        } else {
            None
        }
    }

    /// Reverse this [`Bitboard`], viewing it from the opponent's perspective.
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Rank};
    /// let board = Bitboard::from_rank(Rank::SEVEN);
    /// assert_eq!(board.to_hex_string(), "0x00FF000000000000");
    ///
    /// let flipped = board.flipped();
    /// assert_eq!(flipped.to_hex_string(), "0x000000000000FF00");
    /// ```
    pub const fn flipped(&self) -> Self {
        Self(self.0.swap_bytes())
    }

    /// If `color` is Black, flips this [`Bitboard`].
    /// If `color` is White, does nothing.
    ///
    /// See [`Bitboard::flipped`] for more.
    ///
    /// # Example
    /// ```
    /// # use types::{Color, Bitboard};
    /// assert_eq!(Bitboard::RANK_2.relative_to(Color::White), Bitboard::RANK_2);
    /// assert_eq!(Bitboard::RANK_2.relative_to(Color::Black), Bitboard::RANK_7);
    /// ```
    pub const fn relative_to(self, color: Color) -> Self {
        match color {
            Color::White => self,
            Color::Black => self.flipped(),
        }
    }

    /// Checks if this [`Bitboard`] is empty, or all zeros.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let board = Bitboard::new(0x0);
    /// assert!(board.is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Checks if this [`Bitboard`] is NOT empty, or contains at least one `1`.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let board = Bitboard::CORNERS;
    /// assert!(board.is_nonempty());
    /// ```
    pub const fn is_nonempty(&self) -> bool {
        self.0 != 0
    }

    /// Checks if this [`Bitboard`] contains any of the bits within `other`.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let rank_1 = Bitboard::RANK_1;
    /// let rank_5 = Bitboard::RANK_5;
    /// let file_a = Bitboard::FILE_A;
    /// assert_eq!(rank_1.contains(&file_a), true);
    /// assert_eq!(rank_1.contains(&rank_5), false);
    /// ```
    pub const fn contains(&self, other: &Self) -> bool {
        self.0 & other.0 != 0
    }

    /*
    /// Returns `true` if `self` contains every bit set in `other`.
    pub const fn contains_all(&self, other: &Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` if `self` contains none bits set in `other`.
    pub const fn contains_none(&self, other: &Self) -> bool {
        self.0 & other.0 != 0
    }
     */

    /// Toggles the bit corresponding to the location of the provided [`Square`] to `1` (on).
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let mut board = Bitboard::default();
    /// board.set(Square::G2);
    /// assert_eq!(board.to_hex_string(), "0x0000000000004000");
    /// ```
    pub fn set(&mut self, square: Square) {
        self.set_index(square.index());
    }

    /// Gets the value of the bit corresponding to the location of the provided [`Square`].
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let board = Bitboard::FILE_A;
    /// assert!(board.get(Square::A3));
    /// ```
    pub const fn get(&self, square: Square) -> bool {
        self.get_index(square.index())
    }

    /// Toggles the bit corresponding to the location of the provided [`Square`] to `0` (off).
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Square};
    /// let mut board = Bitboard::RANK_1;
    /// board.clear(Square::C1);
    /// assert_eq!(board.to_hex_string(), "0x00000000000000FB");
    /// ```
    pub fn clear(&mut self, square: Square) {
        self.clear_index(square.index())
    }

    /// Remove all squares from `other` in `self`
    pub fn remove(&mut self, other: &Self) {
        self.0 &= !other.0
    }

    /// Returns the index of the lowest non-zero bit of this [`Bitboard`], as a [`Square`].
    ///
    /// If `self` is empty, this yields `None`,
    pub fn lsb(&self) -> Option<Square> {
        self.is_nonempty()
            .then(|| Square(self.0.trailing_zeros() as u8))
    }

    /// Returns the index of the lowest non-zero bit of this [`Bitboard`], as a [`Square`].
    ///
    /// It is undefined behavior to call this function when `self` is empty.
    pub fn lsb_unchecked(&self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Pops and returns the index of the lowest non-zero bit of this [`Bitboard`], as a [`Square`].
    ///
    /// If `self` is empty, this yields `None`,
    pub fn pop_lsb(&mut self) -> Option<Square> {
        let lsb = self.lsb();
        self.clear_lsb();
        lsb
    }

    /// Clears the lowest non-zero bit from `self`, if there is a square to clear.
    pub fn clear_lsb(&mut self) {
        self.0 &= self.0.wrapping_sub(1);
    }

    /// Toggles the bit corresponding to the specified [`Square`].
    pub fn toggle_square(&mut self, square: Square) {
        *self ^= Self::from_square(square);
    }

    /// Toggles the bit at `index`.
    pub fn toggle_index(&mut self, index: usize) {
        *self ^= Self::from_index(index);
    }

    /// Sets the bit at `index` to `1`.
    ///
    /// # Panics
    /// If `index > 63` with debug assertions enabled.
    fn set_index(&mut self, index: usize) {
        debug_assert!(index < 64, "Index must be between [0,64)");
        // self.0 |= 1 << index;
        *self |= Self::from_index(index);
    }

    /// Returns `true` if the bit at `index` is set, else `false`.
    ///
    /// # Panics
    /// If `index > 63` with debug assertions enabled.
    const fn get_index(&self, index: usize) -> bool {
        debug_assert!(index < 64, "Index must be between [0,64)");
        (self.0 & 1 << index) != 0
    }

    /// Sets the bit at `index` to `0`.
    ///
    /// # Panics
    /// If `index > 63` with debug assertions enabled.
    fn clear_index(&mut self, index: usize) {
        debug_assert!(index < 64, "Index must be between [0,64)");
        // self.0 ^= 1 << index;
        *self ^= Self::from_index(index);
    }

    /// Returns a [`BitboardIter`] to iterate over all of the set bits as [`Square`]s.
    pub const fn iter(&self) -> BitboardIter {
        BitboardIter { bitboard: *self }
    }

    /// Returns a [`BitboardSubsetIter`] to iterate over all of the subsets of this bitboard.
    pub const fn subsets(&self) -> BitboardSubsetIter {
        BitboardSubsetIter {
            bitboard: *self,
            subset: Self::EMPTY_BOARD,
            remaining: 2usize.pow(self.population() as u32),
        }
    }

    /// Yields the total number of `1`s in this [`Bitboard`].
    ///
    /// In other words, this function determines how many bits are activated.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// let board = Bitboard::RANK_1;
    /// assert_eq!(board.population(), 8);
    /// ```
    pub const fn population(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Shifts this [`Bitboard`] forward by `n`, according to `color`.
    ///
    /// If `color` is White, this shifts `n` ranks up. If Black, it shifts by `n` rank down.
    ///
    /// Note: This can "wrap" by advancing beyond the end of the board, so be careful!
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    /// let rank4 = Bitboard::RANK_4;
    /// assert_eq!(rank4.advance_by(Color::White, 1), Bitboard::RANK_5);
    /// assert_eq!(rank4.advance_by(Color::Black, 1), Bitboard::RANK_3);
    /// // Wrapping
    /// assert_eq!(rank4.advance_by(Color::White, 5), Bitboard::RANK_1);
    /// ```
    pub const fn advance_by(self, color: Color, n: u32) -> Self {
        // Black magic: If `color` is White, this rotates left by 8, which is the same as "n ranks up"
        // If `color` is Black, this rotates left by 496, which is the same as rotating right by 8, or "n ranks down"
        Self(self.0.rotate_left(n * 8 * (1 + color as u32 * 62)))
    }

    /// Shifts this [`Bitboard`] backward by `n`, according to `color`.
    ///
    /// If `color` is White, this shifts `n` ranks up. If Black, it shifts by `n` ranks down.
    ///
    /// Note: This can "wrap" by advancing beyond the end of the board, so be careful!
    ///
    /// # Example
    /// ```
    /// # use types::{Bitboard, Color};
    /// let rank4 = Bitboard::RANK_4;
    /// assert_eq!(rank4.retreat_by(Color::White, 1), Bitboard::RANK_3);
    /// assert_eq!(rank4.retreat_by(Color::Black, 1), Bitboard::RANK_5);
    /// // Wrapping
    /// assert_eq!(rank4.retreat_by(Color::Black, 5), Bitboard::RANK_1);
    /// ```
    pub const fn retreat_by(self, color: Color, n: u32) -> Self {
        // Black magic: If `color` is White, this rotates right by 8, which is the same as "n ranks down"
        // If `color` is Black, this rotates right by 496, which is the same as rotating left by 8, or "n ranks up"
        Self(self.0.rotate_right(n * 8 * (1 + color as u32 * 62)))
    }

    /// Shifts this [`Bitboard`] by one rank up.
    ///
    /// If already at the final rank (8), returns an empty board.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// assert_eq!(Bitboard::RANK_4.north(), Bitboard::RANK_5);
    /// assert_eq!(Bitboard::RANK_8.north(), Bitboard::EMPTY_BOARD);
    /// ```
    pub const fn north(self) -> Self {
        Self(self.0 << 8)
    }

    // Rank down
    /// Shifts this [`Bitboard`] by one rank board.
    ///
    /// If already at the first rank (1), returns an empty board.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// assert_eq!(Bitboard::RANK_4.south(), Bitboard::RANK_3);
    /// assert_eq!(Bitboard::RANK_1.south(), Bitboard::EMPTY_BOARD);
    /// ```
    pub const fn south(self) -> Self {
        Self(self.0 >> 8)
    }

    /// Shifts this [`Bitboard`] by one [`File`] up.
    ///
    /// If already at the first file (a), returns an empty board.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// assert_eq!(Bitboard::FILE_C.east(), Bitboard::FILE_D);
    /// assert_eq!(Bitboard::FILE_H.east(), Bitboard::EMPTY_BOARD);
    /// ```
    pub const fn east(self) -> Self {
        // Post-shift mask
        Self((self.0 << 1) & Self::NOT_FILE_A.0)
    }

    /// Shifts this [`Bitboard`] by one [`File`] down.
    ///
    /// If already at the final file (h), returns an empty board.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// assert_eq!(Bitboard::FILE_C.west(), Bitboard::FILE_B);
    /// assert_eq!(Bitboard::FILE_A.west(), Bitboard::EMPTY_BOARD);
    /// ```
    pub const fn west(self) -> Self {
        // Post-shift mask
        Self((self.0 >> 1) & Self::NOT_FILE_H.0)
    }

    /// Combination of [`Bitboard::north`] and [`Bitboard::east`].
    ///
    /// If already at the edge, returns an empty board.
    ///
    /// This operation is faster than calling [`Bitboard::north`] and [`Bitboard::east`] separately.
    pub const fn northeast(self) -> Self {
        // Post-shift mask
        Self((self.0 << 9) & Self::NOT_FILE_A.0)
    }

    /// Combination of [`Bitboard::south`] and [`Bitboard::east`].
    ///
    /// If already at the edge, returns an empty board.
    ///
    /// This operation is faster than calling [`Bitboard::south`] and [`Bitboard::east`] separately.
    pub const fn southeast(self) -> Self {
        // Post-shift mask
        Self((self.0 >> 7) & Self::NOT_FILE_A.0)
    }

    /// Combination of [`Bitboard::north`] and [`Bitboard::west`].
    ///
    /// If already at the edge, returns an empty board.
    ///
    /// This operation is faster than calling [`Bitboard::north`] and [`Bitboard::west`] separately.
    pub const fn northwest(self) -> Self {
        // Post-shift mask
        Self((self.0 << 7) & Self::NOT_FILE_H.0)
    }

    /// Combination of [`Bitboard::south`] and [`Bitboard::west`].
    ///
    /// If already at the edge, returns an empty board.
    ///
    /// This operation is faster than calling [`Bitboard::south`] and [`Bitboard::west`] separately.
    pub const fn southwest(self) -> Self {
        // Post-shift mask
        Self((self.0 >> 9) & Self::NOT_FILE_H.0)
    }

    /*
    /// Creates a mask of all squares in front of `square` (according to `color`) that are either directly in front or on the adjacent files.
    pub fn passed_pawn_mask(square: Square, color: Color) -> Self {
        let rank = match color {
            Color::White => square.rank().increase(),
            Color::Black => square.rank().decrease(),
        };

        let forward_ranks = Self::FULL_BOARD << rank;

        let file = square.file();
        let left = Self::from(file.decrease());
        let right = Self::from(file.increase());
        let center = Self::from_file(file);

        let files = center | left | right;

        forward_ranks & files
    }
     */

    /// `const` analog of [`std::ops::BitAnd::bitand`].
    ///
    /// Returns the bitwise AND (`&`) of `self` and `other`.
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// `const` analog of [`std::ops::BitOr::bitor`].
    ///
    /// Returns the bitwise OR (`|`) of `self` and `other`.
    pub const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// `const` analog of [`std::ops::BitXor::bitxor`].
    ///
    /// Returns the bitwise XOR (`^`) of `self` and `other`.
    pub const fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// `const` analog of [`Not::not`].
    ///
    /// Returns the logical negation (`!`) of `self`, flipping all `1`'s to `0`'s and vice versa.
    pub const fn not(self) -> Self {
        Self(!self.0)
    }

    /// Formats this [`Bitboard`] as a hexadecimal string.
    pub fn to_hex_string(&self) -> String {
        format!("0x{:0>16X}", self.0)
    }
}

impl FromStr for Bitboard {
    type Err = anyhow::Error;
    /// Constructs a new [`Bitboard`] from the provided string.
    ///
    /// The string may be a binary or hexadecimal number, and may be proceeded with `0b` or `0x`.
    ///
    /// # Example
    /// ```
    /// # use types::Bitboard;
    /// use std::str::FromStr;
    /// let board1 = Bitboard::from_str("0x00FF000000000000").unwrap();
    /// let board2 = Bitboard::from_str("00FF000000000000").unwrap();
    /// let board3 = Bitboard::from_str("0000000011111111000000000000000000000000000000000000000000000000").unwrap();
    /// let board4 = Bitboard::from_str("0b0000000011111111000000000000000000000000000000000000000000000000").unwrap();
    /// assert_eq!(board1, board2);
    /// assert_eq!(board1, board3);
    /// assert_eq!(board1, board4);
    /// assert_eq!(board1.to_hex_string(), "0x00FF000000000000");
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let bits = s.to_lowercase();

        if bits.len() == 64 || bits.len() == 66 {
            let bits = bits.trim_start_matches("0b");
            let bits = u64::from_str_radix(bits, 2).map_err(|_| {
                anyhow!("Invalid Bitboard string: Expected binary digits, got {bits}")
            })?;
            Ok(Self::new(bits))
        } else if bits.len() == 16 || bits.len() == 18 {
            let bits = bits.trim_start_matches("0x");
            let bits = u64::from_str_radix(bits, 16).map_err(|_| {
                anyhow!("Invalid Bitboard string: Expected hexadecimal digits, got {bits}")
            })?;
            Ok(Self::new(bits))
        } else {
            bail!("Invalid Bitboard string: Invalid length {}. Length must be either 64 (binary) or 16 (hexadecimal)", bits.len())
        }
    }
}

macro_rules! impl_bitwise_op {
    // Impl op and op_assign for Self
    ($op:tt, $op_assign:tt, $func:ident, $func_assign:ident) => {
        impl std::ops::$op for Bitboard {
            type Output = Self;
            fn $func(self, rhs: Self) -> Self::Output {
                Self(self.0.$func(rhs.0))
            }
        }

        impl std::ops::$op_assign for Bitboard {
            fn $func_assign(&mut self, rhs: Self) {
                self.0.$func_assign(rhs.0);
            }
        }

        impl std::ops::$op<Square> for Bitboard {
            type Output = Self;
            fn $func(self, rhs: Square) -> Self::Output {
                self.$func(rhs.bitboard())
            }
        }

        impl std::ops::$op_assign<Square> for Bitboard {
            fn $func_assign(&mut self, rhs: Square) {
                self.$func_assign(rhs.bitboard());
            }
        }
    };
}

impl_bitwise_op!(BitAnd, BitAndAssign, bitand, bitand_assign);
impl_bitwise_op!(BitOr, BitOrAssign, bitor, bitor_assign);
impl_bitwise_op!(BitXor, BitXorAssign, bitxor, bitxor_assign);
impl_bitwise_op!(Shl, ShlAssign, shl, shl_assign);
impl_bitwise_op!(Shr, ShrAssign, shr, shr_assign);

impl Not for Bitboard {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Shl<File> for Bitboard {
    type Output = Self;
    fn shl(self, rhs: File) -> Self::Output {
        Self::new(self.0 << rhs.0)
    }
}

impl Shr<File> for Bitboard {
    type Output = Self;
    fn shr(self, rhs: File) -> Self::Output {
        Self::new(self.0 >> rhs.0)
    }
}

impl Shl<Rank> for Bitboard {
    type Output = Self;
    fn shl(self, rhs: Rank) -> Self::Output {
        Self::new(self.0 << (rhs.0 * 8))
    }
}

impl Shr<Rank> for Bitboard {
    type Output = Self;
    fn shr(self, rhs: Rank) -> Self::Output {
        Self::new(self.0 >> (rhs.0 * 8))
    }
}

impl Index<Square> for Bitboard {
    type Output = bool;
    /// A [`Bitboard`] can be indexed by a [`Square`] to yield `true` or `false`, if the bit at that index is set.
    fn index(&self, index: Square) -> &Self::Output {
        if self.get_index(index.index()) {
            &true
        } else {
            &false
        }
    }
}

impl Index<usize> for Bitboard {
    type Output = bool;
    /// A [`Bitboard`] can be indexed by a [`usize`] to yield `true` or `false`, if the bit at that index is set.
    ///
    /// # Panics
    /// If debug assertions are enabled and `index > 63`.
    fn index(&self, index: usize) -> &Self::Output {
        if self.get_index(index) {
            &true
        } else {
            &false
        }
    }
}

impl<T> From<Option<T>> for Bitboard
where
    Self: From<T>,
{
    /// If `value` is `None`, this yields an empty [`Bitboard`].
    fn from(value: Option<T>) -> Self {
        Self::from_option(value)
    }
}

impl From<Square> for Bitboard {
    fn from(value: Square) -> Self {
        Self::from_square(value)
    }
}

impl From<File> for Bitboard {
    fn from(value: File) -> Self {
        Self::from_file(value)
    }
}

impl From<Rank> for Bitboard {
    fn from(value: Rank) -> Self {
        Self::from_rank(value)
    }
}

impl From<u64> for Bitboard {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<bool> for Bitboard {
    fn from(value: bool) -> Self {
        Self::from_bool(value)
    }
}

impl fmt::UpperHex for Bitboard {
    /// Formats this [`Bitboard`] as a 16-character uppercase hexadecimal string, not including the `0x` prefix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0X{:0>16X}", self.0)
    }
}

impl fmt::LowerHex for Bitboard {
    /// Formats this [`Bitboard`] as a 16-character lowercase hexadecimal string, not including the `0x` prefix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:0>16x}", self.0)
    }
}

impl fmt::Binary for Bitboard {
    /// Formats this [`Bitboard`] as a 64-character binary string, not including the `0b` prefix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0b{:0>64b}", self.0)
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Allocate just enough capacity
        let mut board = String::with_capacity(136);

        for rank in Rank::iter().rev() {
            for file in File::iter() {
                let square = Square::new(file, rank);
                let occupant = if self.get(square) { 'X' } else { '.' };

                board += &format!("{occupant} ");
            }
            board += "\n";
        }

        write!(f, "{board}")
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Allocate just enough capacity
        let mut board = String::with_capacity(198);

        for rank in Rank::iter().rev() {
            board += &format!("{rank}| ");

            for file in File::iter() {
                let square = Square::new(file, rank);
                let occupant = if self.get(square) { 'X' } else { '.' };

                board += &format!("{occupant} ");
            }
            board += "\n";
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

pub struct BitboardIter {
    bitboard: Bitboard,
}

impl Iterator for BitboardIter {
    type Item = Square;
    fn next(&mut self) -> Option<Self::Item> {
        self.bitboard.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.bitboard.population() as usize;
        (size, Some(size))
    }
}

impl ExactSizeIterator for BitboardIter {
    fn len(&self) -> usize {
        self.bitboard.population() as usize
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;
    fn into_iter(self) -> Self::IntoIter {
        BitboardIter { bitboard: self }
    }
}

impl IntoIterator for &Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;
    fn into_iter(self) -> Self::IntoIter {
        BitboardIter { bitboard: *self }
    }
}

impl IntoIterator for &mut Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;
    fn into_iter(self) -> Self::IntoIter {
        BitboardIter { bitboard: *self }
    }
}

/// An iterator over all possible subsets of a [`Bitboard`].
///
/// See [`Bitboard::subsets`].
///
/// This is primarily used in magic bitboard generation, but may also be useful for other purposes, so it is made public.
pub struct BitboardSubsetIter {
    /// The original bitboard whose subsets to iterate over.
    bitboard: Bitboard,

    /// The current subset, which will be the result of `.next()`.
    subset: Bitboard,

    /// How many subsets we have left to iterate.
    remaining: usize,
}

impl Iterator for BitboardSubsetIter {
    type Item = Bitboard;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            // By saving and returning the original subset, we make the iterator return
            // an empty set as the first element and the full subset as the last.
            let subset = self.subset;

            // Performs a Carry-Rippler operation: https://www.chessprogramming.org/Traversing_Subsets_of_a_Set#All_Subsets_of_any_Set
            self.subset.0 = self.subset.0.wrapping_sub(self.bitboard.0) & self.bitboard.0;
            self.remaining -= 1;

            Some(subset)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for BitboardSubsetIter {
    fn len(&self) -> usize {
        self.remaining
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bitboard_to_string() {
        let expected = ". . . . . . . X \n\
                              . . . . . . X . \n\
                              . . . . . X . . \n\
                              . . . . X . . . \n\
                              . . . X . . . . \n\
                              . . X . . . . . \n\
                              . X . . . . . . \n\
                              X . . . . . . . \n";
        assert_eq!(Bitboard::A1_H8_DIAG.to_string(), expected);

        let expected = ". . . . . . . . \n\
                              . . . . . . . . \n\
                              . . . . . . . . \n\
                              . . . . . . . . \n\
                              . . . . . . . . \n\
                              . . . . . . . . \n\
                              X X X X X X X X \n\
                              . . . . . . . . \n";
        assert_eq!(Bitboard::RANK_2.to_string(), expected);

        let board = Bitboard::RANK_2 | Bitboard::FILE_C;
        let expected = ". . X . . . . . \n\
                              . . X . . . . . \n\
                              . . X . . . . . \n\
                              . . X . . . . . \n\
                              . . X . . . . . \n\
                              . . X . . . . . \n\
                              X X X X X X X X \n\
                              . . X . . . . . \n";
        assert_eq!(board.to_string(), expected);
    }

    #[test]
    fn test_bitboard_masking() {
        let file_a = Bitboard::FILE_A;
        let full_board = Bitboard::FULL_BOARD;
        let expected = Bitboard::NOT_FILE_A;

        assert_eq!(file_a ^ full_board, expected);
    }

    #[test]
    fn test_bitboard_from_str() {
        let bits = "0x0101010101010101";
        let board = Bitboard::from_str(&bits).unwrap();
        assert_eq!(board, Bitboard::FILE_A);

        let bits = "0101010101010101";
        let board = Bitboard::from_str(&bits).unwrap();
        assert_eq!(board, Bitboard::FILE_A);

        let bits = "0b0000000100000001000000010000000100000001000000010000000100000001";
        let board = Bitboard::from_str(&bits).unwrap();
        assert_eq!(board, Bitboard::FILE_A);

        let bits = "0000000100000001000000010000000100000001000000010000000100000001";
        let board = Bitboard::from_str(&bits).unwrap();
        assert_eq!(board, Bitboard::FILE_A);

        let bits = "0b0000000200000002000000020000000200000002000000010000000100000001";
        let board = Bitboard::from_str(bits);
        assert!(board.is_err());

        let bits = "0000000200000002000000020000000200000002000000010000000100000001";
        let board = Bitboard::from_str(bits);
        assert!(board.is_err());

        let bits = "x0awdk";
        let board = Bitboard::from_str(bits);
        assert!(board.is_err());

        let bits = "";
        let board = Bitboard::from_str(bits);
        assert!(board.is_err());
    }

    #[test]
    fn test_bitboard_constructors() {
        assert_eq!(Bitboard::RANK_4, Bitboard::from_rank(Rank::FOUR));
    }
}
