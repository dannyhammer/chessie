/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::Instant;

use clap::Parser;

use chessie::{perft, splitperft, Game, Move};

/// Compute total number of states reachable from a position, given a depth.
#[derive(Debug, Parser)]
struct Cli {
    /// Depth to run the perft.
    depth: usize,

    /// The FEN string of then position to run the perft.
    #[arg(required = false)]
    fen: Option<String>,

    /// List of moves to apply to the position before running the perft.
    #[arg(required = false)]
    moves: Vec<String>,

    /// If set, perform a splitperft, displaying the number of nodes reachable after each move available from the root.
    #[arg(short, long, default_value = "false")]
    split: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    // Parse args appropriately
    let mut game = if let Some(fen) = &args.fen {
        Game::from_fen(fen)?
    } else {
        Game::default()
    };

    // Apply moves, if any were provided
    for mv_str in args.moves {
        // Parse move string and apply it
        let mv = Move::from_uci(&game, &mv_str)?;
        game.make_move(mv);
    }

    println!(
        "Computing PERFT({}) of the following position:\n{}\n",
        args.depth,
        game.to_fen()
    );

    let now = Instant::now();
    let total_nodes = if args.split {
        let nodes = splitperft(&game, args.depth);
        println!("\n{nodes}\n");
        nodes
    } else {
        perft(&game, args.depth)
    };

    let elapsed = now.elapsed();

    // Compute nodes-per-second metrics
    let nps = total_nodes as f32 / elapsed.as_secs_f32();
    let m_nps = nps / 1_000_000.0;

    println!("  Total Nodes:\t{total_nodes}");
    println!(" Elapsed Time:\t{elapsed:.1?}");
    println!("  Nodes / Sec:\t{nps:.0}");
    println!("M Nodes / Sec:\t{m_nps:.1}");

    Ok(())
}
