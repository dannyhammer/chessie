use std::{
    // fmt,
    // ops::{Add, AddAssign},
    time::Instant,
};

use super::Game;

/*
/// A result from a perft function.
#[derive(Default, Debug, Clone, Copy)]
pub struct PerftResult {
    /// Depth searched
    depth: usize,

    /// Number of game states reachable.
    nodes: u64,

    /// Number of captures possible.
    captures: u64,

    /// Number of times en passant can be performed.
    eps: u64,

    /// Number of times castling can occur.
    castles: u64,

    /// Number of times a pawn can be promoted. A single move counts as one promotion, not the four possible promotions.
    promotions: u64,

    /// Number of checks that can occur.
    checks: u64,

    /// Number of discovery checks possible.
    discovery_checks: u64,

    /// Number of double checks that can occur.
    double_checks: u64,

    /// Number of checkmates that can occur.
    checkmates: u64,
}

impl Add for PerftResult {
    type Output = Self;
    /// Convenience implementation to add perft results.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            depth: self.depth,
            nodes: self.nodes + rhs.nodes,
            captures: self.captures + rhs.captures,
            eps: self.eps + rhs.eps,
            castles: self.castles + rhs.castles,
            promotions: self.promotions + rhs.promotions,
            checks: self.checks + rhs.checks,
            discovery_checks: self.discovery_checks + rhs.discovery_checks,
            double_checks: self.double_checks + rhs.double_checks,
            checkmates: self.checkmates + rhs.checkmates,
        }
    }
}

impl AddAssign for PerftResult {
    /// Convenience implementation to add perft results.
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl fmt::Display for PerftResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entry = |val, width| format!("{val:>width$}");

        let format_line =
            |depth, nodes, captures, eps, castles, promos, checks, discoveries, doubles, mates| {
                format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    entry(depth, self.depth.to_string().len()),
                    entry(nodes, self.nodes.to_string().len().max("Nodes".len())),
                    entry(captures, self.captures.to_string().len().max("Capt.".len())),
                    entry(eps, self.eps.to_string().len().max("E.p.".len())),
                    entry(castles, self.castles.to_string().len().max("Cast.".len())),
                    entry(
                        promos,
                        self.promotions.to_string().len().max("Promo.".len())
                    ),
                    entry(checks, self.checks.to_string().len().max("Checks".len())),
                    entry(
                        discoveries,
                        self.discovery_checks.to_string().len().max("Disc.".len())
                    ),
                    entry(
                        doubles,
                        self.double_checks.to_string().len().max("Dbl.".len())
                    ),
                    entry(mates, self.checkmates.to_string().len().max("Mates".len()))
                )
            };

        let mut result = format_line(
            String::from(""),
            String::from("Nodes"),
            String::from("Capt."),
            String::from("E.p."),
            String::from("Cast."),
            String::from("Promo."),
            String::from("Checks"),
            String::from("Disc."),
            String::from("Dbl."),
            String::from("Mates"),
        );
        let divider = "-".repeat(result.len());
        result += "\n";
        result += &divider;
        result += "\n";

        for i in 0..self.depth {
            let line = format_line(
                i.to_string(),
                self.nodes.to_string(),
                self.captures.to_string(),
                self.eps.to_string(),
                self.castles.to_string(),
                self.promotions.to_string(),
                self.checkmates.to_string(),
                self.discovery_checks.to_string(),
                self.double_checks.to_string(),
                self.checkmates.to_string(),
            );
            result += &line;
            result += "\n";
        }

        write!(f, "{result}")
    }
}
 */

/// Prints a perft at the specified depth.
///
/// If the generic parameter `SPLIT` is `true`, this will perform a `splitperft`,
/// printing all moves at the first level (`depth`) followed by how many nodes
/// were reached after each of those moves.
///
/// If the generic parameter `PRETTY` is `true`, additional info will be printed.
pub fn print_perft<const PRETTY: bool, const SPLIT: bool>(game: &Game, depth: usize) {
    if PRETTY {
        println!("Computing PERFT({depth}) of the following position:\n{game}\n",);
    }

    let now = Instant::now();
    let total_nodes = if SPLIT {
        let mut total_nodes = 0;
        for mv in game.iter() {
            let nodes = perft(&game.with_move_made(mv), depth - 1);

            println!("{mv}\t{nodes}");
            total_nodes += nodes;
        }
        println!(); // Empty line between last splitperft and total_nodes
        total_nodes
    } else {
        perft(game, depth)
    };

    if PRETTY {
        let elapsed = now.elapsed();

        // Compute nodes-per-second metrics
        let nps = total_nodes as f32 / elapsed.as_secs_f32();
        let m_nps = nps / 1_000_000.0;

        println!("Elapsed Time:          {elapsed:.1?}");
        println!("Total Nodes:           {total_nodes}");
        println!("Nodes / Sec:           {nps:.0}");
        println!("M Nodes / Sec:         {m_nps:.1}");
    } else {
        println!("{total_nodes}");
    }
}

/*
/// Perform a perft at the specified depth, collecting data on captures, castling, promotions, etc.
pub fn perft_full(position: &Position, depth: usize) -> PerftResult {
    let mut res = PerftResult::default();

    if depth == 0 {
        res.nodes = 1;
        // res.captures = 32 - position.bitboards().occupied().population() as u64; // TODO: Fetch original number of pieces
        // res.castles = position.times_castled() as u64;
        // res.checks = position.is_check() as u64;
        // res.checkmates = position.is_checkmate() as u64;
        return res;
    }

    let game = Game::new(position);

    if depth == 1 {
        res.nodes = game.legal_moves().len() as u64;
        // TODO: Functions for `num_captures_available()` etc.
        return res;
    }

    for mv in game {
        let new_pos = position.clone().with_move_made(mv);
        res += perft_full(&new_pos, depth - 1);
    }

    res
}
 */

/// Perform a perft at the specified depth, collecting only data about the number of possible positions (nodes).
///
/// This performs bulk counting, meaning that, at depth 1, it returns the number of available moves,
/// rather than making them, recursing again, and returning 1 for each terminal case.
pub fn perft(game: &Game, depth: usize) -> u64 {
    // Bulk counting; no need to recurse again just to apply a singular move and return 1.
    if depth == 1 {
        let moves = game.get_legal_moves();
        return moves.len() as u64;
    }
    // Recursion limit; return 1, since we're fathoming this node.
    else if depth == 0 {
        return 1;
    }

    // Recursively accumulate the nodes from the remaining depths
    game.iter().fold(0, |nodes, mv| {
        nodes + perft(&game.with_move_made(mv), depth - 1)
    })
}
