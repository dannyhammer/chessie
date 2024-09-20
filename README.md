# Chessie

A (work-in-progress) fast chess library suitable for use in chess engines.

## Overview

This library provides a clean, easy-to-use API for creating and working with chess games.
It supports [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) (FEN) strings for creating positions, as well as [Universal Chess Interface](https://en.wikipedia.org/wiki/Universal_Chess_Interface) (UCI) notation for pieces, squares, moves, and more.

One major goal of mine for this project was to include [documentation tests](https://doc.rust-lang.org/rust-by-example/testing/doc_testing.html) for every function.
As a result, nearly every function in this library has examples of how to use them that double as unit tests.

## Examples

Only generate moves from specific squares (Knights, in this case):

```rust
let game = Game::default(); // Default starting position
let mask = game.knights(Color::White);
for mv in game.iter().only_moves_from(mask) {
    print!("{mv} ");
}
// b1a3 b1c3 g1f3 g1h3
```

Only generate moves that capture enemy pieces:

```rust
let game = Game::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
for mv in game.iter().only_captures() {
    print!("{mv} ");
}
// e2a6 g2h3 f3h3 f3f6 d5e6 e5g6 e5d7 e5f7
```

## Features

Several state-of-the-art chess programming ideas have been incorporated into this library, as well as several Rust-specific paradigms.

### Current

-   User-friendly, heavily-documented, easy-to-read, safe API.
-   (Almost) every function has examples in its documentation.
-   Ability to create, modify, and read chess positions.
-   Compact representation of primitive types such as squares (8 bits) and moves (16 bits).
-   Move generation API through Rust's [`Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html) trait, and allow generation of moves to/from specific squares (such as only generating moves for pawns, or only generating captures).
-   [Bitboards](https://www.chessprogramming.org/Bitboards) for piece layout and move generation.
-   [Magic Bitboards](https://www.chessprogramming.org/Magic_Bitboards) for sliding piece move generation.
-   And many more that I may have neglected to mention

### Future

-   Support for parsing/writing [Extended Position Description](https://www.chessprogramming.org/Extended_Position_Description) (EPD).
-   Support for parsing/writing [Portable Game Notation](https://en.wikipedia.org/wiki/Portable_Game_Notation) (PGN).
-   Support for parsing/writing [Standard Algebraic Notation](<https://en.wikipedia.org/wiki/Algebraic_notation_(chess)>) (SAN).
-   [Chess960](https://www.chessprogramming.org/Chess960) support.
-   Support for other variants, like [Horde Chess](https://www.chess.com/terms/horde-chess).
-   Proper support for un-making moves.
-   General optimizations. See the "Issues" tab for more.

## Acknowledgements:

Special thanks in particular to:

-   [Sebastian Lague](https://www.youtube.com/@SebastianLague), for his [chess programming series](https://www.youtube.com/watch?v=_vqlIPDR2TU&list=PLFt_AvWsXl0cvHyu32ajwh2qU1i6hl77c) on YouTube that ultimate inspired me to do this project.
-   The [Chess Programming Wiki](https://www.chessprogramming.org/), and all those who contribute to free, open-source knowledge.
-   The folks over at the [Engine Programming Discord](https://discord.com/invite/F6W6mMsTGN), for their patience with my silly questions and invaluable help overall.
-   [Analog-Hors](https://github.com/analog-hors), for an excellent [article on magic bitboards](https://analog-hors.github.io/site/magic-bitboards/)
