use std::path::Path;

use super::{Bitboard, Color, Rank, Square};

use anyhow::{bail, Result};
use rand::random;

/// FEN string for the starting position of chess.
pub const FEN_STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A popular FEN string for debugging move generation.
pub const FEN_KIWIPETE: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// Maximum possible number of moves in a given chess position.
///
/// Found [here](<https://www.chessprogramming.org/Chess_Position#cite_note-4>)
pub const MAX_NUM_MOVES: usize = 218;

/// Number of possible combinations of castling rights.
///
/// Used for Zobrist hashing.
pub const NUM_CASTLING_RIGHTS: usize = 16;

/// Deltas for the movement of the Queen.
pub const QUEEN_DELTAS: [(i8, i8); 8] = [
    /* Rook */
    (1, 0),
    (0, -1),
    (-1, 0),
    (0, 1),
    /* Bishop */
    (1, 1),
    (1, -1),
    (-1, -1),
    (-1, 1),
];

/// Deltas for the movement of the Rook.
pub const ROOK_DELTAS: [(i8, i8); 4] = [
    QUEEN_DELTAS[0],
    QUEEN_DELTAS[1],
    QUEEN_DELTAS[2],
    QUEEN_DELTAS[3],
];

/// Deltas for the movement of the Bishop.
pub const BISHOP_DELTAS: [(i8, i8); 4] = [
    QUEEN_DELTAS[4],
    QUEEN_DELTAS[5],
    QUEEN_DELTAS[6],
    QUEEN_DELTAS[7],
];

/// Deltas for the movement of the Knight.
pub const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (1, -2),
    (2, 1),
    (2, -1),
    (-1, 2),
    (-1, -2),
    (-2, 1),
    (-2, -1),
];

/// Generates `.dat` files for ray tables.
///
/// See `ray_containing` and others for more details in `brogle_core`.
pub fn generate_ray_table_datfiles<P: AsRef<Path>>(outdir: P) -> std::io::Result<()> {
    // 64 * 64 * 8 = 32,768
    // 2D Bitboard array being cast to u8 (8 u8 in a u64)

    // Generate the blobs
    let ray_between_exclusive: [u8; 32_768] =
        unsafe { std::mem::transmute(generate_ray_between_table()) };
    let ray_containing: [u8; 32_768] =
        unsafe { std::mem::transmute(generate_ray_containing_table()) };

    // Write the blobs
    let path = |name| Path::new(outdir.as_ref()).join(name);

    std::fs::write(path("ray_between.dat"), ray_between_exclusive)?;
    std::fs::write(path("ray_containing.dat"), ray_containing)?;

    Ok(())
}

fn generate_ray_between_table() -> [[Bitboard; Square::COUNT]; Square::COUNT] {
    let mut rays = [[Bitboard::EMPTY_BOARD; Square::COUNT]; Square::COUNT];

    for from in Square::iter() {
        for (df, dr) in QUEEN_DELTAS {
            let mut ray = Bitboard::default(); // Do not include `from`
            let mut to = from;
            while let Some(shifted) = to.offset(df, dr) {
                ray.set(shifted);
                to = shifted;
                rays[from][to] = ray ^ to.bitboard(); // Do not include `to`
            }
        }
    }

    rays
}

fn generate_ray_containing_table() -> [[Bitboard; Square::COUNT]; Square::COUNT] {
    let mut rays = [[Bitboard::EMPTY_BOARD; Square::COUNT]; Square::COUNT];

    for from in Square::iter() {
        let fr = from.rank();
        let ff = from.file();
        for to in Square::iter() {
            let tr = to.rank();
            let tf = to.file();
            if from == to {
                rays[from][to] = Bitboard::from_square(from);
            } else if fr == tr {
                rays[from][to] = Bitboard::from_rank(fr);
            } else if ff == tf {
                rays[from][to] = Bitboard::from_file(ff);
            } else {
                // To check if these lie on a diagonal, compute (y1 - y2) / (x1 - x2)
                let file_diff = from.file().inner() as i32 - to.file().inner() as i32;
                let rank_diff = from.rank().inner() as i32 - to.rank().inner() as i32;

                // Checked division
                let diff = if rank_diff != 0 {
                    file_diff / rank_diff
                } else {
                    0
                };

                if diff == 1 {
                    rays[from][to] |= from.bitboard();
                    // I'm too lazy to figure out the proper math, so I'm just going to cast rays in the diagonals
                    let mut tmp = from;
                    // First ray goes Northeast
                    while let Some(shifted) = tmp.offset(1, 1) {
                        rays[from][to] |= shifted.bitboard();
                        tmp = shifted;
                    }
                    tmp = from;
                    // Second ray goes Southwest
                    // I'm intentionally not resetting tmp here, so that the square for `from` gets OR'd in
                    while let Some(shifted) = tmp.offset(-1, -1) {
                        rays[from][to] |= shifted.bitboard();
                        tmp = shifted;
                    }
                } else if diff == -1 {
                    rays[from][to] |= from.bitboard();
                    // Do it again, in the Northwest / Southeast directions
                    let mut tmp = from;
                    while let Some(shifted) = tmp.offset(-1, 1) {
                        rays[from][to] |= shifted.bitboard();
                        tmp = shifted;
                    }
                    tmp = from;
                    while let Some(shifted) = tmp.offset(1, -1) {
                        rays[from][to] |= shifted.bitboard();
                        tmp = shifted;
                    }
                }
            }
        }
    }

    rays
}

/// Generates the default mobility for each of the pieces of standard chess, and writes the mobility to new files created in `outdir`.
///
/// This will produce the following 9 files located in `outdir`:
///     * `rook_mobility.blob`
///     * `bishop_mobility.blob`
///     * `knight_mobility.blob`
///     * `king_mobility.blob`
///     * `white_pawn_push_mobility.blob`
///     * `black_pawn_push_mobility.blob`
///     * `white_pawn_attack_mobility.blob`
///     * `black_pawn_attack_mobility.blob`
///
/// You can use `include_bytes!()` to read from these blobs like so:
/// ```compile_fail
/// const KNIGHT_MOVES: [Bitboard; 64] = unsafe { std::mem::transmute(*include_bytes!("knight_mobility.blob")) };
/// ```
pub fn generate_piece_attack_datfiles<P: AsRef<Path>>(outdir: P) -> std::io::Result<()> {
    // Generate the blobs
    let bishop: [u8; 512] = unsafe { std::mem::transmute(generate_bishop_mobility()) };
    let rook: [u8; 512] = unsafe { std::mem::transmute(generate_rook_mobility()) };
    let knight: [u8; 512] = unsafe { std::mem::transmute(generate_knight_mobility()) };
    let king: [u8; 512] = unsafe { std::mem::transmute(generate_king_mobility()) };
    let wpp: [u8; 512] = unsafe { std::mem::transmute(generate_pawn_pushes(Color::White)) };
    let bpp: [u8; 512] = unsafe { std::mem::transmute(generate_pawn_pushes(Color::Black)) };
    let wpa: [u8; 512] = unsafe { std::mem::transmute(generate_pawn_attacks(Color::White)) };
    let bpa: [u8; 512] = unsafe { std::mem::transmute(generate_pawn_attacks(Color::Black)) };

    // Write the blobs
    let path = |name| Path::new(outdir.as_ref()).join(name);
    std::fs::write(path("bishop_attacks.dat"), bishop)?;
    std::fs::write(path("rook_attacks.dat"), rook)?;
    std::fs::write(path("knight_attacks.dat"), knight)?;
    std::fs::write(path("king_attacks.dat"), king)?;
    std::fs::write(path("white_pawn_pushes.dat"), wpp)?;
    std::fs::write(path("black_pawn_pushes.dat"), bpp)?;
    std::fs::write(path("white_pawn_attacks.dat"), wpa)?;
    std::fs::write(path("black_pawn_attacks.dat"), bpa)?;

    Ok(())
}

/// Generates the default push mobility for Pawns.
///
/// Pawns, by default, may push forward by one, except when pushing from their starting rank (rank 2 for White, rank 7 for Black), in which case they may push forward by two.
fn generate_pawn_pushes(color: Color) -> [Bitboard; 64] {
    let mut boards = [Bitboard::default(); Square::COUNT];
    for square in Square::iter() {
        let bb = Bitboard::from_square(square);

        if square.rank() == Rank::second(color) {
            boards[square] = bb.advance_by(color, 1) | bb.advance_by(color, 2);
        } else {
            boards[square] = bb.advance_by(color, 1);
        }
    }
    boards
}

/// Generates the default attack mobility for Pawns.
///
/// Pawns, by default, may capture diagonally forward by one.
fn generate_pawn_attacks(color: Color) -> [Bitboard; 64] {
    let mut boards = [Bitboard::default(); Square::COUNT];
    for square in Square::iter() {
        let bb = Bitboard::from_square(square);

        boards[square] = bb.advance_by(color, 1).east() | bb.advance_by(color, 1).west();
    }
    boards
}

/// Generates the moves from every location for the "Leaper" pieces.
/// Leapers may "leap" or "jump" to a square a specified distance away.
///
/// In standard chess, the Leapers are the King and Knight.
fn generate_leaper_mobility(deltas: &[(i8, i8)]) -> [Bitboard; Square::COUNT] {
    // Represents all locations this piece can reach from that square/index.
    let mut mobility = [Bitboard::default(); Square::COUNT];

    for square in Square::iter() {
        // All reachable locations from `square`.
        // This is empty because we cannot "move to" the square where we are currently.
        let mut movement = Bitboard::default();

        // Loop over every pair of deltas
        for (df, dr) in deltas {
            // If shifting this location by the delta results in a valid position, add it to the movement mask.
            if let Some(shifted) = square.offset(*df, *dr) {
                movement.set(shifted);
            }
        }

        // Store the mobility from this square.
        mobility[square] = movement;
    }

    mobility
}

/// Generates the moves from every location for the "Rider" pieces.
/// Riders may "ride" or "slide" an unlimited number of squares in a direction.
///
/// In standard chess, the Riders are the Rook, Bishop, and Queen.
fn generate_rider_mobility(deltas: &[(i8, i8)]) -> [Bitboard; Square::COUNT] {
    // Represents all locations this piece can reach from that square/index.
    let mut mobility = [Bitboard::default(); Square::COUNT];

    for square in Square::iter() {
        // All reachable locations from `square`.
        // This is empty because we cannot "move to" the square where we are currently.
        let mut movement = Bitboard::default();

        // Loop over every pair of deltas
        for (df, dr) in deltas {
            // Create a "ray" that represents movement in this direction.
            let mut ray = square;

            // Shift the ray and append it's movement, until we reach the end of the board.
            while let Some(shifted) = ray.offset(*df, *dr) {
                movement.set(shifted);
                ray = shifted;
            }
        }

        // Store the mobility from this square.
        mobility[square] = movement;
    }

    mobility
}

/// Generates the default mobility for the King.
fn generate_king_mobility() -> [Bitboard; 64] {
    generate_leaper_mobility(&QUEEN_DELTAS)
}

/// Generates the default mobility for the Knight.
fn generate_knight_mobility() -> [Bitboard; 64] {
    generate_leaper_mobility(&KNIGHT_DELTAS)
}

/// Generates the default mobility for the Rook.
fn generate_rook_mobility() -> [Bitboard; Square::COUNT] {
    generate_rider_mobility(&ROOK_DELTAS)
}

/// Generates the default mobility for the Bishop.
fn generate_bishop_mobility() -> [Bitboard; Square::COUNT] {
    generate_rider_mobility(&BISHOP_DELTAS)
}

/*
/// Generates the default mobility for the Queen.
fn generate_queen_mobility() -> [Bitboard; Square::COUNT] {
    generate_rider_mobility(&QUEEN_DELTAS)
}

/// Generates the default mobility for the Dragon (Queen + Knight).
fn generate_dragon_mobility() -> [Bitboard; Square::COUNT] {
    let mut dragon = generate_rider_mobility(&QUEEN_DELTAS);
    let knight = generate_leaper_mobility(&KNIGHT_DELTAS);

    for square in Square::iter() {
        dragon[square] |= knight[square];
    }

    dragon
}
 */

/// Generate magics for the Bishop and Rook and store them in `<outdir>/rook_magics.rs` and `<outdir>/bishop_magics.rs`.
pub fn generate_magics<P: AsRef<Path>>(outdir: P) -> std::io::Result<()> {
    let rook_magics = find_and_write_magics(&ROOK_DELTAS, "ROOK");
    let rook_magic_path = Path::new(outdir.as_ref()).join("rook_magics.rs");
    std::fs::write(rook_magic_path, rook_magics)?;

    let bishop_magics = find_and_write_magics(&BISHOP_DELTAS, "BISHOP");
    let bishop_magic_path = Path::new(outdir.as_ref()).join("bishop_magics.rs");
    std::fs::write(bishop_magic_path, bishop_magics)?;

    Ok(())
}

/// Find all magics for the provided piece and format them as a `String` to be written to a file.
fn find_and_write_magics(deltas: &[(i8, i8)], piece_name: &str) -> String {
    let mut s = format!(
        "pub const {piece_name}_MAGICS: &[MagicBitboardData; {}] = &[\n",
        Square::COUNT
    );

    let mut table_size = 0;

    for square in Square::iter() {
        let index_bits = compute_blockers(deltas, square).population();
        let (entry, table) = find_magic(deltas, square, index_bits);

        s += format!(
            "  MagicBitboardData {{ blockers: 0x{:016X}, magic: 0x{:016X}, shift: {}, offset: {} }},\n",
            entry.blockers, entry.magic, entry.shift, table_size
        )
        .as_str();

        table_size += table.len();
    }

    s += "];\n";
    s += format!("pub const {piece_name}_TABLE_SIZE: usize = {table_size};").as_str();

    s
}

#[derive(Debug, Default, Clone, Copy)]
struct MagicBitboardData {
    blockers: Bitboard,
    magic: u64,
    shift: u8,
}

/// Obtain the appropriate index into a magic bitboard table for the given blocker bitboard.
fn magic_index(data: &MagicBitboardData, blockers: Bitboard) -> usize {
    let blockers = blockers & data.blockers;
    let hash = blockers.inner().wrapping_mul(data.magic);
    (hash >> data.shift) as usize
}

/// Computes a [`Bitboard`] of containing all possible blocker squares for a sliding
/// piece (whose movement is defined by `deltas`) at `square`.
///
/// This is the same as computing the default movement for a sliding piece at `square`
/// and XOR'ing to remove the squares on the edges of the board.
fn compute_blockers(deltas: &[(i8, i8)], square: Square) -> Bitboard {
    let mut blockers = Bitboard::EMPTY_BOARD;

    // Loop over the directions this piece can move
    for (df, dr) in deltas {
        // Start with the present square
        let mut ray = square;
        // If we can continue moving in this direction, add it to the blockers bitboard
        // The order here matters- we don't care about the edges of the board.
        while let Some(shifted) = ray.offset(*df, *dr) {
            blockers |= ray.bitboard();
            ray = shifted;
        }
    }

    // The starting square cannot be a blocker
    blockers ^ square.bitboard()
}

/// Computes a [`Bitboard`] of containing all squares that can be attacked by a
/// sliding piece (whose movement is defined by `deltas`) at `square`, taking into
/// account occupied squares through `blockers`.
///
/// The computed [`Bitboard`] will allow the slider at `square` to move to the first
/// blocked square in `blockers`, so filtering out friendly pieces should be handled
/// elsewhere.
fn compute_blocked_attacks(deltas: &[(i8, i8)], square: Square, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::EMPTY_BOARD;

    // Loop over the directions this piece can move
    for (df, dr) in deltas {
        let mut ray = square;

        // Loop until we encounter the first occupied square
        while !blockers.get(ray) {
            // If we have not moved off the edge of the board, add this square to the attack bitboard
            if let Some(shifted) = ray.offset(*df, *dr) {
                ray = shifted;
                attacks |= ray.bitboard();
            } else {
                // If we HAVE moved off the edge of the board, we can exit the loop and check the next delta
                break;
            }
        }
    }

    attacks
}

/// Finds a valid magic for `square`.
fn find_magic(
    deltas: &[(i8, i8)],
    square: Square,
    index_bits: u8,
) -> (MagicBitboardData, Vec<Bitboard>) {
    let blockers = compute_blockers(deltas, square);
    let shift = 64 - index_bits;

    loop {
        // Only a few bits are needed, so generate a random number with only a few bits set
        let magic = random::<u64>() & random::<u64>() & random::<u64>();

        let magic_data = MagicBitboardData {
            blockers,
            magic,
            shift,
        };

        if let Ok(table) = try_magic(deltas, square, &magic_data) {
            return (magic_data, table);
        }
    }
}

/// Attempts to use the provided magic to generate valid movegen for all possible blockers.
fn try_magic(
    deltas: &[(i8, i8)],
    square: Square,
    magic_data: &MagicBitboardData,
) -> Result<Vec<Bitboard>> {
    let index_bits = 64 - magic_data.shift;
    let mut table = vec![Bitboard::EMPTY_BOARD; 1 << index_bits];

    // We need to check if the table will be valid for every possible configuration of blockers
    for blockers in magic_data.blockers.subsets() {
        let attacks = compute_blocked_attacks(deltas, square, blockers);
        let entry = &mut table[magic_index(magic_data, blockers)];

        // If the entry is empty, we can fill it
        if entry.is_empty() {
            *entry = attacks;
        } else if *entry != attacks {
            // Two entries map to the same slot; hash collision
            bail!("Hash collision between {entry} and {attacks}");
        }
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ROOK_DELTAS;

    #[test]
    fn test_compute_blockers_bitboard() {
        let rook_e4_blockers = compute_blockers(&ROOK_DELTAS, Square::E4);
        assert_eq!(
            rook_e4_blockers.to_string(),
            ". . . . . . . . 
. . . . X . . . 
. . . . X . . . 
. . . . X . . . 
. X X X . X X . 
. . . . X . . . 
. . . . X . . . 
. . . . . . . . 
"
        );
    }

    #[test]
    fn test_compute_moves_bitboard() {
        // Rook at E4.
        // Blockers at E1, C4, F4, E7.
        let blockers = Bitboard(4503600231350288);
        let moves = compute_blocked_attacks(&ROOK_DELTAS, Square::E4, blockers);
        assert_eq!(
            moves.to_string(),
            ". . . . . . . . 
. . . . X . . . 
. . . . X . . . 
. . . . X . . . 
. . X X . X . . 
. . . . X . . . 
. . . . X . . . 
. . . . X . . . 
"
        );
    }
}
