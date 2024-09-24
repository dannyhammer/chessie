use anyhow::Context;
use chessie::{print_perft, Game, Move};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Print usage if insufficient arguments provided
    if args.len() < 2 {
        println!("Usage: {} <depth> [fen] [moves]", args[0]);
        std::process::exit(1);
    }

    // Parse args appropriately
    let depth = args[1].parse().context(format!(
        "Failed to parse {:?} as depth value. Expected integer.",
        args[1]
    ))?;
    let mut game = if let Some(fen) = args.get(2) {
        Game::from_fen(fen)?
    } else {
        Game::default()
    };

    // Apply moves, if any were provided
    if args.len() > 3 {
        for mv_str in args[3].split_ascii_whitespace() {
            // Parse move string and apply it
            let mv = Move::from_uci(&game, mv_str)?;
            game.make_move(mv);
        }
    }

    print_perft::<false, true>(&game, depth);

    Ok(())
}
