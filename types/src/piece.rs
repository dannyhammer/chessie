use std::{
    fmt,
    ops::{Index, IndexMut, Neg},
    str::FromStr,
};

use anyhow::{bail, Result};

/// Represents the color of a player, piece, square, etc. within a chess board.
///
/// In Western chess, White traditionally moves first, and therefore [`Color`] defaults to [`Color::White`].
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Color {
    #[default]
    White,
    Black,
}

impl Color {
    /// Number of color variants.
    pub const COUNT: usize = 2;

    /// An array of both colors, starting with White.
    pub const fn all() -> [Self; Self::COUNT] {
        [Self::White, Self::Black]
    }

    /// An iterator over both colors, starting with White.
    pub fn iter() -> impl Iterator<Item = Self> {
        Self::all().into_iter()
    }

    /// Creates a new [`Color`] from a set of bits.
    ///
    /// `bits` must be `[0,1]`.
    ///
    /// # Panics
    /// If `bits` is greater than `1`.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// let white = Color::from_bits(0);
    /// assert!(white.is_ok());
    /// assert_eq!(white.unwrap(), Color::White);
    ///
    /// let err = Color::from_bits(42);
    /// assert!(err.is_err());
    /// ```
    pub fn from_bits(bits: u8) -> Result<Self> {
        match bits {
            0 => Ok(Self::White),
            1 => Ok(Self::Black),
            _ => bail!("Invalid bits for Color: Bits must be between [0,1]. Got {bits}."),
        }
    }

    /// Creates a new [`Color`] from a set of bits, ignoring safety checks.
    ///
    /// `bits` must be `[0,1]`.
    ///
    /// # Panics
    /// If `bits` is greater than `1` and debug assertions are enabled.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// let white = Color::from_bits(0);
    /// assert!(white.is_ok());
    /// assert_eq!(white.unwrap(), Color::White);
    ///
    /// let err = Color::from_bits(42);
    /// assert!(err.is_err());
    /// ```
    pub const fn from_bits_unchecked(bits: u8) -> Self {
        debug_assert!(
            bits <= 1,
            "Invalid bits for Color: Bits must be between [0,1]"
        );

        // Safety: Since `Color` is a `repr(u8)` enum, we can cast safely here.
        unsafe { std::mem::transmute(bits) }
    }

    /// Creates a new [`Color`] from a `bool`, where `false = White`.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// let white = Color::from_bool(false);
    /// assert_eq!(white, Color::White);
    ///
    /// let black = Color::from_bool(true);
    /// assert_eq!(black, Color::Black);
    /// ```
    pub const fn from_bool(color: bool) -> Self {
        Self::from_bits_unchecked(color as u8)
    }

    /// Returns `true` if this [`Color`] is White.
    pub const fn is_white(&self) -> bool {
        *self as u8 & 1 == 0
    }

    /// Returns `true` if this [`Color`] is Black.
    pub const fn is_black(&self) -> bool {
        *self as u8 & 1 != 0
    }

    /// Returns a multiplier for negating numbers relative to this color.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.negation_multiplier(), 1);
    /// assert_eq!(Color::Black.negation_multiplier(), -1);
    /// ```
    pub const fn negation_multiplier(&self) -> i8 {
        // TODO: Which of these 3 is faster?

        // A: Match
        match self {
            Self::White => 1,
            Self::Black => -1,
        }

        // B: Shift
        // 1 - ((*self as i8) << 1)

        // C: Bitwise
        // (*self as i8 ^ 1) - (*self as i8 & 1)
    }

    /// Returns this [`Color`]'s opposite / inverse / enemy.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.opponent(), Color::Black);
    /// assert_eq!(Color::Black.opponent(), Color::White);
    /// ```
    pub const fn opponent(&self) -> Self {
        Self::from_bits_unchecked(self.bits() ^ 1)
    }

    /// Returns this [`Color`] as a `usize`.
    ///
    /// Will be `0` for White, `1` for Black.
    ///
    /// Useful for indexing into lists.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.index(), 0);
    /// assert_eq!(Color::Black.index(), 1);
    /// ```
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Returns this [`Color`] as a `u8`.
    ///
    /// Will be `0` for White, `1` for Black.
    ///
    /// Useful for bit twiddling.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.bits(), 0);
    /// assert_eq!(Color::Black.bits(), 1);
    /// ```
    pub const fn bits(&self) -> u8 {
        *self as u8
    }

    /// Creates a [`Color`] from a `char`, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// let white = Color::from_uci('w');
    /// assert!(white.is_ok());
    /// assert_eq!(white.unwrap(), Color::White);
    ///
    /// let err = Color::from_uci('x');
    /// assert!(err.is_err());
    /// ```
    pub fn from_uci(color: char) -> Result<Self> {
        match color {
            'w' | 'W' => Ok(Self::White),
            'b' | 'B' => Ok(Self::Black),
            _ => bail!("Color must be either 'w' or 'b' (case-insensitive). Found {color}"),
        }
    }

    /// Creates a [`Color`] based on the ASCII case of the provided character, with uppercase being White and lowercase being Black.
    ///
    /// Note this is intended to follow the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation, but can be used in odd ways, such as trying to find the color of the char `'z'` (Black).
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::from_case('k'), Color::Black);
    /// ```
    pub const fn from_case(c: char) -> Self {
        Self::from_bool(c.is_ascii_lowercase())
    }

    /// Converts this [`Color`] to a char, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.to_uci(), 'w');
    /// ```
    pub const fn to_uci(&self) -> char {
        match self {
            Self::White => 'w',
            Self::Black => 'b',
        }
    }

    /// Converts this [`Color`] to a `str`, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// assert_eq!(Color::White.as_str(), "w");
    /// ```
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::White => "w",
            Self::Black => "b",
        }
    }

    /// Fetches a human-readable name for this [`Color`].
    ///
    /// # Example
    /// ```
    /// # use types::Color;
    /// let white = Color::White;
    /// assert_eq!(white.name(), "white");
    /// ```
    pub const fn name(&self) -> &'static str {
        match self {
            Self::White => "white",
            Self::Black => "black",
        }
    }
}

impl Neg for Color {
    type Output = Self;
    /// Negating [`Color::White`] yields [`Color::Black`] and vice versa.
    fn neg(self) -> Self::Output {
        self.opponent()
    }
}

/// Represents the kind (or "class") that a chess piece can be.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    /// Number of piece variants.
    pub const COUNT: usize = 6;

    /// An array of all 6 [`PieceKind`]s.
    ///
    /// In the order: `Pawn`, `Knight`, `Bishop`, `Rook`, `Queen`, `King`.
    pub const fn all() -> [Self; Self::COUNT] {
        use PieceKind::*;
        [Pawn, Knight, Bishop, Rook, Queen, King]
    }

    /// An array of 5 non-King [`PieceKind`]s.
    ///
    /// In the order: `Pawn`, `Knight`, `Bishop`, `Rook`, `Queen`.
    pub const fn all_except_king() -> [Self; Self::COUNT - 1] {
        use PieceKind::*;
        [Pawn, Knight, Bishop, Rook, Queen]
    }

    /// An iterator over all [`PieceKind`]s, starting with Pawn.
    pub fn iter() -> impl Iterator<Item = Self> {
        Self::all().into_iter()
    }

    /// Creates a new [`PieceKind`] from a set of bits.
    ///
    /// `bits` must be `[0,5]`.
    ///
    /// # Panics
    /// If `bits` is greater than `5`.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::from_bits(4);
    /// assert!(queen.is_ok());
    /// assert_eq!(queen.unwrap(), PieceKind::Queen);
    ///
    /// let err = PieceKind::from_bits(42);
    /// assert!(err.is_err());
    /// ```
    pub fn from_bits(bits: u8) -> Result<Self> {
        match bits {
            0 => Ok(Self::Pawn),
            1 => Ok(Self::Knight),
            2 => Ok(Self::Bishop),
            3 => Ok(Self::Rook),
            4 => Ok(Self::Queen),
            5 => Ok(Self::King),
            _ => bail!("Invalid bits for PieceKind: Bits must be between [0,5]. Got {bits}."),
        }
    }

    /// Creates a new [`PieceKind`] from a set of bits, ignoring safety checks.
    ///
    /// `bits` must be `[0,5]`.
    ///
    /// # Panics
    /// If `bits` is greater than `5` when debug assertions are enabled.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::from_bits_unchecked(4);
    /// assert_eq!(queen, PieceKind::Queen);
    /// ```
    pub const fn from_bits_unchecked(bits: u8) -> Self {
        debug_assert!(
            bits <= 5,
            "Invalid bits for PieceKind: Bits must be between [0,5]"
        );

        // Safety: Since `PieceKind` is a `repr(u8)` enum, we can cast safely here.
        unsafe { std::mem::transmute(bits) }
    }

    /// Fetches the internal bit value of this [`PieceKind`].
    ///
    /// Will always be `[0,5]`.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let bits = PieceKind::Queen.bits();
    /// assert_eq!(bits, 4);
    /// ```
    pub const fn bits(&self) -> u8 {
        *self as u8
    }

    /// Returns this [`PieceKind`] as a `usize`.
    ///
    /// Useful for indexing into lists.
    ///
    /// Will always be `[0,5]`.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let index = PieceKind::Queen.index();
    /// assert_eq!(index, 4);
    /// ```
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Creates a new [`PieceKind`] from a character, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// Will return a [`anyhow::Error`] if `kind` is not a valid character.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::from_uci('Q');
    /// assert!(queen.is_ok());
    /// assert_eq!(queen.unwrap(), PieceKind::Queen);
    /// ```
    pub fn from_uci(kind: char) -> Result<Self> {
        match kind {
            'P' | 'p' => Ok(Self::Pawn),
            'N' | 'n' => Ok(Self::Knight),
            'B' | 'b' => Ok(Self::Bishop),
            'R' | 'r' => Ok(Self::Rook),
            'Q' | 'q' => Ok(Self::Queen),
            'K' | 'k' => Ok(Self::King),
            _ => bail!("Invalid char for PieceKind: Got {kind}."),
        }
    }

    /// Fetches a human-readable name for this [`PieceKind`].
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::Queen;
    /// assert_eq!(queen.name(), "queen");
    /// ```
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Pawn => "pawn",
            Self::Knight => "knight",
            Self::Bishop => "bishop",
            Self::Rook => "rook",
            Self::Queen => "queen",
            Self::King => "king",
        }
    }

    /// Converts this [`PieceKind`] to a character, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// Will always be a lowercase letter.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::Queen;
    /// assert_eq!(queen.to_uci(), 'q');
    /// ```
    pub const fn to_uci(&self) -> char {
        match self {
            Self::Pawn => 'p',
            Self::Knight => 'n',
            Self::Bishop => 'b',
            Self::Rook => 'r',
            Self::Queen => 'q',
            Self::King => 'k',
        }
    }

    /// Alias for [`PieceKind::to_uci`].
    pub const fn char(&self) -> char {
        self.to_uci()
    }

    /// Converts this [`PieceKind`] to a `str`, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// Will always be a lowercase letter.
    ///
    /// # Example
    /// ```
    /// # use types::PieceKind;
    /// let queen = PieceKind::Queen;
    /// assert_eq!(queen.as_str(), "q");
    /// ```
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pawn => "p",
            Self::Knight => "n",
            Self::Bishop => "b",
            Self::Rook => "r",
            Self::Queen => "q",
            Self::King => "k",
        }
    }
}

/// Represents a chess piece on the game board.
///
/// Internally, this is represented as a `u8` with the following bit pattern:
///
/// ```text
///     0000 0 000
///      |   |  |
///      |   |  +- Represents the PieceKind.
///      |   +- Represents the Color. `0` for White, `1` for Black.
///      +- Unused.
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Piece(u8);

impl Piece {
    pub const WHITE_PAWN: Self = Self::new(Color::White, PieceKind::Pawn);
    pub const WHITE_ROOK: Self = Self::new(Color::White, PieceKind::Rook);
    pub const WHITE_KING: Self = Self::new(Color::White, PieceKind::King);
    pub const WHITE_QUEEN: Self = Self::new(Color::White, PieceKind::Queen);
    pub const WHITE_KNIGHT: Self = Self::new(Color::White, PieceKind::Knight);
    pub const WHITE_BISHOP: Self = Self::new(Color::White, PieceKind::Bishop);

    pub const BLACK_PAWN: Self = Self::new(Color::Black, PieceKind::Pawn);
    pub const BLACK_ROOK: Self = Self::new(Color::Black, PieceKind::Rook);
    pub const BLACK_KING: Self = Self::new(Color::Black, PieceKind::King);
    pub const BLACK_QUEEN: Self = Self::new(Color::Black, PieceKind::Queen);
    pub const BLACK_KNIGHT: Self = Self::new(Color::Black, PieceKind::Knight);
    pub const BLACK_BISHOP: Self = Self::new(Color::Black, PieceKind::Bishop);

    /// Number of unique piece variants.
    pub const COUNT: usize = Color::COUNT * PieceKind::COUNT;

    /// Mask for the color bits.
    const COLOR_MASK: u8 = 0b0000_1000;
    /// Start index of color bits.
    const COLOR_BITS: u8 = 3;

    /// Creates a new [`Piece`] from the given [`Color`] and [`PieceKind`].
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color, PieceKind};
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// assert_eq!(white_knight.to_string(), "N");
    /// ```
    pub const fn new(color: Color, kind: PieceKind) -> Self {
        // 0000 0000 => white
        // 0000 1000 => black
        Self(color.bits() << Self::COLOR_BITS | kind.bits())
    }

    /// Fetches the [`Color`] of this [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color, PieceKind};
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// assert_eq!(white_knight.color(), Color::White);
    /// ```
    pub const fn color(&self) -> Color {
        Color::from_bits_unchecked(self.0 >> Self::COLOR_BITS)
    }

    /// Returns `true` if this [`Piece`]'s [`Color`] is White.
    ///
    /// # Example
    /// ```
    /// # use types::Piece;
    /// assert!(Piece::WHITE_KNIGHT.is_white());
    /// assert!(!Piece::BLACK_KNIGHT.is_white());
    /// ```
    pub const fn is_white(&self) -> bool {
        self.0 >> Self::COLOR_BITS == 0
    }

    /// Returns `true` if this [`Piece`]'s [`Color`] is Black.
    ///
    /// # Example
    /// ```
    /// # use types::Piece;
    /// assert!(Piece::BLACK_KNIGHT.is_black());
    /// assert!(!Piece::WHITE_KNIGHT.is_black());
    /// ```
    pub const fn is_black(&self) -> bool {
        self.0 >> Self::COLOR_BITS != 0
    }

    /// Fetches the [`PieceKind`] of this [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color, PieceKind};
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// assert_eq!(white_knight.kind(), PieceKind::Knight);
    /// ```
    pub const fn kind(&self) -> PieceKind {
        // Clear the color bit
        PieceKind::from_bits_unchecked(self.0 & !Self::COLOR_MASK)
    }

    /// Returns `true` if this [`Piece`] is a Pawn.
    pub const fn is_pawn(&self) -> bool {
        matches!(self.kind(), PieceKind::Pawn)
    }

    /// Returns `true` if this [`Piece`] is a Knight.
    pub const fn is_knight(&self) -> bool {
        matches!(self.kind(), PieceKind::Knight)
    }

    /// Returns `true` if this [`Piece`] is a Bishop.
    pub const fn is_bishop(&self) -> bool {
        matches!(self.kind(), PieceKind::Bishop)
    }

    /// Returns `true` if this [`Piece`] is a Rook.
    pub const fn is_rook(&self) -> bool {
        matches!(self.kind(), PieceKind::Rook)
    }

    /// Returns `true` if this [`Piece`] is a Queen.
    pub const fn is_queen(&self) -> bool {
        matches!(self.kind(), PieceKind::Queen)
    }

    /// Returns `true` if this [`Piece`] is a King.
    pub const fn is_king(&self) -> bool {
        matches!(self.kind(), PieceKind::King)
    }

    /// Returns `true` if this [`Piece`] is a slider (Rook, Bishop, Queen).
    pub const fn is_slider(&self) -> bool {
        matches!(
            self.kind(),
            PieceKind::Queen | PieceKind::Rook | PieceKind::Bishop
        )
    }

    /// Returns `true` if this [`Piece`] is an orthogonal slider (Rook, Queen).
    pub const fn is_orthogonal_slider(&self) -> bool {
        matches!(self.kind(), PieceKind::Queen | PieceKind::Rook)
    }

    /// Returns `true` if this [`Piece`] is a diagonal slider (Bishop, Queen).
    pub const fn is_diagonal_slider(&self) -> bool {
        matches!(self.kind(), PieceKind::Queen | PieceKind::Bishop)
    }

    /// Fetches the [`Color`] and [`PieceKind`] of this [`Piece`].
    pub const fn parts(&self) -> (Color, PieceKind) {
        (self.color(), self.kind())
    }

    /// Returns the index value of this [`Piece`], as a `usize`.
    ///
    /// Useful for indexing into lists of size 6 or 12.
    pub const fn index(&self) -> usize {
        let offset = if self.is_white() {
            0
        } else {
            PieceKind::all().len()
        };

        self.kind().bits() as usize + offset
    }

    /// Creates a new [`Piece`] from a character, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// Will return a [`anyhow::Error`] if `piece` is not a valid character.
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color, PieceKind};
    /// let white_knight = Piece::from_uci('N').unwrap();
    /// assert_eq!(white_knight.color(), Color::White);
    /// assert_eq!(white_knight.kind(), PieceKind::Knight);
    /// ```
    pub fn from_uci(piece: char) -> Result<Self> {
        let kind = PieceKind::from_uci(piece)?;
        let color = Color::from_case(piece);
        Ok(Self::new(color, kind))
    }

    /// Converts this [`Piece`] into a character, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color, PieceKind};
    /// let white_knight = Piece::new(Color::White, PieceKind::Knight);
    /// assert_eq!(white_knight.to_uci(), 'N');
    /// ```
    pub const fn to_uci(&self) -> char {
        if self.is_white() {
            self.kind().char().to_ascii_uppercase()
        } else {
            self.kind().char().to_ascii_lowercase()
        }
    }

    /// Alias for [`Piece::to_uci`].
    pub const fn char(&self) -> char {
        self.to_uci()
    }

    /// Converts this [`Piece`] to a `str`, according to the [Universal Chess Interface](https://en.wikipedia.org//wiki/Universal_Chess_Interface) notation.
    ///
    /// # Example
    /// ```
    /// # use types::Piece;
    /// assert_eq!(Piece::WHITE_QUEEN.as_str(), "Q");
    /// assert_eq!(Piece::BLACK_PAWN.as_str(), "p");
    /// ```
    pub const fn as_str(&self) -> &'static str {
        match self.color() {
            Color::White => match self.kind() {
                PieceKind::Pawn => "P",
                PieceKind::Knight => "N",
                PieceKind::Bishop => "B",
                PieceKind::Rook => "R",
                PieceKind::Queen => "Q",
                PieceKind::King => "K",
            },
            Color::Black => match self.kind() {
                PieceKind::Pawn => "p",
                PieceKind::Knight => "n",
                PieceKind::Bishop => "b",
                PieceKind::Rook => "r",
                PieceKind::Queen => "q",
                PieceKind::King => "k",
            },
        }
    }

    /// Promotes (or, in a less likely scenario, demotes) this [`Piece`] to a new [`PieceKind`], based on the value of `promotion`, consuming `self` in the process and returning the promoted [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::{Color, Piece, PieceKind};
    /// let pawn = Piece::from_uci('p').unwrap();
    /// assert_eq!(pawn.kind(), PieceKind::Pawn);
    /// assert_eq!(pawn.color(), Color::Black);
    /// let queen = pawn.promoted(PieceKind::Queen);
    /// assert_eq!(queen.kind(), PieceKind::Queen);
    /// assert_eq!(queen.color(), Color::Black);
    /// ```
    pub const fn promoted(self, promotion: PieceKind) -> Self {
        Self::new(self.color(), promotion)
    }

    /// Demotes this [`Piece`] to a Pawn, consuming `self` in the process and returning the demoted [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, PieceKind};
    /// let queen = Piece::from_uci('Q').unwrap();
    /// let pawn = queen.demoted();
    /// assert_eq!(pawn.kind(), PieceKind::Pawn);
    /// ```
    pub const fn demoted(self) -> Self {
        self.promoted(PieceKind::Pawn)
    }

    /// Inverts the [`Color`] of this [`Piece`] to the opponent's color, consuming `self` and returning the new [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::{Piece, Color};
    /// let mut king = Piece::from_uci('k').unwrap();
    /// let michael_jackson = king.inverted();
    /// assert_eq!(michael_jackson.color(), Color::White);
    /// ```
    pub fn inverted(self) -> Self {
        Self::new(self.color().opponent(), self.kind())
    }

    /// Fetches a human-readable name for this [`Piece`].
    ///
    /// # Example
    /// ```
    /// # use types::Piece;
    /// let white_queen = Piece::WHITE_QUEEN;
    /// assert_eq!(white_queen.name(), "white queen");
    /// ```
    pub fn name(&self) -> String {
        format!("{} {}", self.color().name(), self.kind().name())
    }
}

impl<T> Index<Piece> for [T; PieceKind::COUNT] {
    type Output = T;
    /// [`Piece`] can be used to index into a list of six elements.
    fn index(&self, index: Piece) -> &Self::Output {
        &self[index.kind().index()]
    }
}

impl<T> IndexMut<Piece> for [T; PieceKind::COUNT] {
    /// [`Piece`] can be used to index into a list of six elements.
    fn index_mut(&mut self, index: Piece) -> &mut Self::Output {
        &mut self[index.kind().index()]
    }
}

macro_rules! impl_common_traits {
    ($type:ty) => {
        impl<T> Index<$type> for [T; <$type>::COUNT] {
            type Output = T;
            /// [`$type`] can be used to index into a list of [`<$type>::COUNT`] elements.
            fn index(&self, index: $type) -> &Self::Output {
                &self[index.index()]
            }
        }

        impl<T> IndexMut<$type> for [T; <$type>::COUNT] {
            /// [`$type`] can be used to mutably index into a list of [`<$type>::COUNT`] elements.
            fn index_mut(&mut self, index: $type) -> &mut Self::Output {
                &mut self[index.index()]
            }
        }

        impl FromStr for $type {
            type Err = anyhow::Error;
            /// Does the same as [`Self::from_uci`], but only if `s` is one character in length.
            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                if s.len() != 1 {
                    bail!("Invalid str for <$type>: Must be a str of len 1. Got {s:?}");
                }

                Self::from_uci(s.as_bytes()[0] as char)
            }
        }

        impl AsRef<str> for $type {
            /// Alias for [`Self::as_str`].
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $type {
            /// By default, a $type displays as a lowercase char.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_uci())
            }
        }

        impl fmt::Debug for $type {
            /// Debug formatting displays a $type as its human-readable name and index value.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "\"{}\" ({})", self.name(), self.index())
            }
        }
    };
}

impl_common_traits!(Piece);
impl_common_traits!(PieceKind);
impl_common_traits!(Color);
