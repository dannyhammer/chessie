use chessie::{Game, Move, Position};

// Sets up a game from the provided FEN
fn setup_game(fen: &str) -> Position {
    Position::from_fen(fen).unwrap()
}

/// Checks if `moves` and `legal_moves` contain all the same elements, ignoring order
fn lists_match<'a>(
    pos: &Position,
    moves: impl IntoIterator<Item = Move>,
    legal_moves: impl IntoIterator<Item = &'a str>,
) {
    let fen = pos.to_fen();

    let mut moves = moves
        .into_iter()
        .map(|m| format!("{m}"))
        .collect::<Vec<String>>();
    let mut legal_moves = legal_moves
        .into_iter()
        .map(|m| m.to_string())
        .collect::<Vec<String>>();

    moves.sort();
    legal_moves.sort();

    let mut diff = Vec::with_capacity(moves.len());

    let missing_or_extra = if moves.len() > legal_moves.len() {
        for m in &moves {
            if !legal_moves.contains(&m) {
                diff.push(m);
            }
        }
        format!("Extra: {diff:?}")
    } else {
        for m in &legal_moves {
            if !moves.contains(&m) {
                diff.push(m);
            }
        }
        format!("Missing: {diff:?}")
    };

    assert_eq!(
        moves.len(),
        legal_moves.len(),
        "\nMoves: {moves:?}\nLegal: {legal_moves:?}\n{missing_or_extra}\nPosition: {fen}\n{pos}"
    );

    for mv in moves {
        let mv = mv.to_string();
        assert!(
            legal_moves.contains(&mv),
            "\n\tIllegal move {mv}\nPosition: {fen}\n{pos}\n\tLegal moves:\n\t{legal_moves:?}",
        );
    }
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_rnbqkbnr_pppppppp_8_8_8_8_PPPPPPPP_RNBQKBNR_w_KQkq_X_0_1() {
    let pos = setup_game("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4", "f2f3",
        "f2f4", "g2g3", "g2g4", "h2h3", "h2h4", "b1a3", "b1c3", "g1f3", "g1h3",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_r3k2r_p1ppqpb1_bn2pnp1_3PN3_1p2P3_2N2Q1p_PPPBBPPP_R3K2R_w_KQkq_X_0_1() {
    let pos = setup_game("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a2a3", "a2a4", "b2b3", "g2g3", "g2h3", "g2g4", "d5d6", "d5e6", "c3b1", "c3d1", "c3a4",
        "c3b5", "e5d3", "e5c4", "e5g4", "e5c6", "e5g6", "e5d7", "e5f7", "d2c1", "d2e3", "d2f4",
        "d2g5", "d2h6", "e2d1", "e2f1", "e2d3", "e2c4", "e2b5", "e2a6", "a1b1", "a1c1", "a1d1",
        "h1f1", "h1g1", "f3d3", "f3e3", "f3g3", "f3h3", "f3f4", "f3g4", "f3f5", "f3h5", "f3f6",
        "e1c1", "e1d1", "e1f1", "e1g1",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1r4b1_8_4R3_pP6_8_1K1N3q_8_3k4_w_X_a6_0_1() {
    let pos = setup_game("1r4b1/8/4R3/pP6/8/1K1N3q/8/3k4 w - a6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["b5b6", "b3a2", "b3b2", "b3a3", "b3c3", "b3a4", "b3c4"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k2r3b_8_5N2_3N4_1N1K2Nq_8_1N6_b7_w_X_X_0_1() {
    let pos = setup_game("k2r3b/8/5N2/3N4/1N1K2Nq/8/1N6/b7 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "b4a2", "b4c2", "b4d3", "b4a6", "b4c6", "d4c3", "d4d3", "d4e3", "d4c4", "d4e4", "d4c5",
        "d4e5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k2r4_5n1b_6P1_2pP4_8_P2KP2q_2P5_8_w_X_c6_0_1() {
    let pos = setup_game("k2r4/5n1b/6P1/2pP4/8/P2KP2q/2P5/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c2c3", "c2c4", "a3a4", "d5d6", "g6h7", "d3d2", "d3e2", "d3c3", "d3c4", "d3e4",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_7k_r7_8_2pP4_8_8_5K2_8_w_X_c6_0_1() {
    let pos = setup_game("7k/r7/8/2pP4/8/8/5K2/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "d5c6", "f2e1", "f2f1", "f2g1", "f2e2", "f2g2", "f2e3", "f2f3", "f2g3",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_7k_b7_8_2pP4_8_8_5K2_8_w_X_c6_0_1() {
    let pos = setup_game("7k/b7/8/2pP4/8/8/5K2/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "f2e1", "f2f1", "f2g1", "f2e2", "f2g2", "f2e3", "f2f3", "f2g3",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_7k_8_8_b1pP1K2_8_8_8_8_w_X_c6_0_1() {
    let pos = setup_game("7k/8/8/b1pP1K2/8/8/8/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "d5c6", "f5e4", "f5f4", "f5g4", "f5e5", "f5g5", "f5e6", "f5f6", "f5g6",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_7k_8_8_r1pP1K2_8_8_8_8_w_X_c6_0_1() {
    let pos = setup_game("7k/8/8/r1pP1K2/8/8/8/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "f5e4", "f5f4", "f5g4", "f5e5", "f5g5", "f5e6", "f5f6", "f5g6",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_3b3k_8_8_2pP4_8_8_3K4_8_w_X_c6_0_1() {
    let pos = setup_game("3b3k/8/8/2pP4/8/8/3K4/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "d5c6", "d2c1", "d2d1", "d2e1", "d2c2", "d2e2", "d2c3", "d2d3", "d2e3",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_3r3k_8_8_2pP4_8_8_3K4_8_w_X_c6_0_1() {
    let pos = setup_game("3r3k/8/8/2pP4/8/8/3K4/8 w - c6 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "d5d6", "d2c1", "d2d1", "d2e1", "d2c2", "d2e2", "d2c3", "d2d3", "d2e3",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2R2r2_4R3_8_1RK2Rr1_8_8_2R5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2R2r2/4R3/8/1RK2Rr1/8/8/2R5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1c2", "c1c3", "b4b1", "b4b2",
        "b4b3", "b4a4", "b4b5", "b4b6", "b4b7", "b4b8", "e6e1", "e6e2", "e6e3", "e6e4", "e6e5",
        "e6a6", "e6b6", "e6c6", "e6d6", "e6f6", "e6g6", "e6h6", "e6e7", "e6e8", "f4d4", "f4e4",
        "f4g4", "c7c5", "c7c6", "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2R2b2_4R3_8_1RK2Rb1_8_8_2R5_w_X_X_0_1() {
    let pos = setup_game("k1b5/2R2b2/4R3/8/1RK2Rb1/8/8/2R5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1c2", "c1c3", "b4b1", "b4b2",
        "b4b3", "b4a4", "b4b5", "b4b6", "b4b7", "b4b8", "f4f1", "f4f2", "f4f3", "f4d4", "f4e4",
        "f4g4", "f4f5", "f4f6", "f4f7", "c7c5", "c7c6", "c7a7", "c7b7", "c7d7", "c7e7", "c7f7",
        "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2R2q2_4R3_8_1RK2Rq1_8_8_2R5_w_X_X_0_1() {
    let pos = setup_game("k1q5/2R2q2/4R3/8/1RK2Rq1/8/8/2R5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1c2", "c1c3", "b4b1", "b4b2",
        "b4b3", "b4a4", "b4b5", "b4b6", "b4b7", "b4b8", "f4d4", "f4e4", "f4g4", "c7c5", "c7c6",
        "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2R2b2_4R3_8_1RK2Rq1_8_8_2R5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2R2b2/4R3/8/1RK2Rq1/8/8/2R5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1c2", "c1c3", "b4b1", "b4b2",
        "b4b3", "b4a4", "b4b5", "b4b6", "b4b7", "b4b8", "f4d4", "f4e4", "f4g4", "c7c5", "c7c6",
        "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2B2r2_4B3_8_1BK2Br1_8_8_2B5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2B2r2/4B3/8/1BK2Br1/8/8/2B5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1b2", "c1d2", "c1a3", "c1e3", "b4e1", "b4d2", "b4a3", "b4c3", "b4a5", "b4c5", "b4d6",
        "b4e7", "b4f8", "e6g4", "e6d5", "e6f5", "e6d7", "e6f7", "e6c8", "c4b3", "c4c3", "c4d3",
        "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2B2b2_4B3_8_1BK2Bb1_8_8_2B5_w_X_X_0_1() {
    let pos = setup_game("k1b5/2B2b2/4B3/8/1BK2Bb1/8/8/2B5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1b2", "c1d2", "c1a3", "c1e3", "b4e1", "b4d2", "b4a3", "b4c3", "b4a5", "b4c5", "b4d6",
        "b4e7", "b4f8", "f4d2", "f4h2", "f4e3", "f4g3", "f4e5", "f4g5", "f4d6", "f4h6", "c7a5",
        "c7e5", "c7b6", "c7d6", "c7b8", "c7d8", "e6d5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4",
        "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2B2q2_4B3_8_1BK2Bq1_8_8_2B5_w_X_X_0_1() {
    let pos = setup_game("k1q5/2B2q2/4B3/8/1BK2Bq1/8/8/2B5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1b2", "c1d2", "c1a3", "c1e3", "b4e1", "b4d2", "b4a3", "b4c3", "b4a5", "b4c5", "b4d6",
        "b4e7", "b4f8", "e6d5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2B2b2_4B3_8_1BK2Bq1_8_8_2B5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2B2b2/4B3/8/1BK2Bq1/8/8/2B5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1b2", "c1d2", "c1a3", "c1e3", "b4e1", "b4d2", "b4a3", "b4c3", "b4a5", "b4c5", "b4d6",
        "b4e7", "b4f8", "e6d5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2Q2r2_4Q3_8_1QK2Qr1_8_8_2Q5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2Q2r2/4Q3/8/1QK2Qr1/8/8/2Q5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1b2", "c1c2", "c1d2", "c1a3",
        "c1c3", "c1e3", "b4b1", "b4e1", "b4b2", "b4d2", "b4a3", "b4b3", "b4c3", "b4a4", "b4a5",
        "b4b5", "b4c5", "b4b6", "b4d6", "b4b7", "b4e7", "b4b8", "b4f8", "e6e1", "e6e2", "e6e3",
        "e6e4", "e6g4", "e6d5", "e6e5", "e6f5", "e6a6", "e6b6", "e6c6", "e6d6", "e6f6", "e6g6",
        "e6h6", "e6d7", "e6e7", "e6f7", "e6c8", "e6e8", "f4d4", "f4e4", "f4g4", "c7c5", "c7c6",
        "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2Q2b2_4Q3_8_1QK2Qb1_8_8_2Q5_w_X_X_0_1() {
    let pos = setup_game("k1b5/2Q2b2/4Q3/8/1QK2Qb1/8/8/2Q5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1b2", "c1c2", "c1d2", "c1a3",
        "c1c3", "c1e3", "b4b1", "b4e1", "b4b2", "b4d2", "b4a3", "b4b3", "b4c3", "b4a4", "b4a5",
        "b4b5", "b4c5", "b4b6", "b4d6", "b4b7", "b4e7", "b4b8", "b4f8", "f4f1", "f4d2", "f4f2",
        "f4h2", "f4e3", "f4f3", "f4g3", "f4d4", "f4e4", "f4g4", "f4e5", "f4f5", "f4g5", "f4d6",
        "f4f6", "f4h6", "f4f7", "c7a5", "c7c5", "c7e5", "c7b6", "c7c6", "c7d6", "c7a7", "c7b7",
        "c7d7", "c7e7", "c7f7", "c7b8", "c7c8", "c7d8", "e6d5", "e6f7", "c4b3", "c4c3", "c4d3",
        "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2Q2q2_4Q3_8_1QK2Qq1_8_8_2Q5_w_X_X_0_1() {
    let pos = setup_game("k1q5/2Q2q2/4Q3/8/1QK2Qq1/8/8/2Q5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1b2", "c1c2", "c1d2", "c1a3",
        "c1c3", "c1e3", "b4b1", "b4e1", "b4b2", "b4d2", "b4a3", "b4b3", "b4c3", "b4a4", "b4a5",
        "b4b5", "b4c5", "b4b6", "b4d6", "b4b7", "b4e7", "b4b8", "b4f8", "f4d4", "f4e4", "f4g4",
        "e6d5", "e6f7", "c7c5", "c7c6", "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5",
        "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2Q2b2_4Q3_8_1QK2Qq1_8_8_2Q5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2Q2b2/4Q3/8/1QK2Qq1/8/8/2Q5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1a1", "c1b1", "c1d1", "c1e1", "c1f1", "c1g1", "c1h1", "c1b2", "c1c2", "c1d2", "c1a3",
        "c1c3", "c1e3", "b4b1", "b4e1", "b4b2", "b4d2", "b4a3", "b4b3", "b4c3", "b4a4", "b4a5",
        "b4b5", "b4c5", "b4b6", "b4d6", "b4b7", "b4e7", "b4b8", "b4f8", "f4d4", "f4e4", "f4g4",
        "e6d5", "e6f7", "c7c5", "c7c6", "c7c8", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5",
        "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2P2r2_4P3_8_1PK2Pr1_8_8_2P5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2P2r2/4P3/8/1PK2Pr1/8/8/2P5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1c2", "b4b5", "e6e7", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2P2b2_4P3_8_1PK2Pb1_8_8_2P5_w_X_X_0_1() {
    let pos = setup_game("k1b5/2P2b2/4P3/8/1PK2Pb1/8/8/2P5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1c2", "b4b5", "f4f5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2P2q2_4P3_8_1PK2Pq1_8_8_2P5_w_X_X_0_1() {
    let pos = setup_game("k1q5/2P2q2/4P3/8/1PK2Pq1/8/8/2P5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1c2", "b4b5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2P2b2_4P3_8_1PK2Pq1_8_8_2P5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2P2b2/4P3/8/1PK2Pq1/8/8/2P5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c1c2", "b4b5", "e6f7", "c4b3", "c4c3", "c4d3", "c4d4", "c4b5", "c4c5", "c4d5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2p2r2_4p3_8_1pK2pr1_8_8_2p5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2p2r2/4p3/8/1pK2pr1/8/8/2p5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4d3", "c4b4", "c4d4", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2p2b2_4p3_8_1pK2pb1_8_8_2p5_w_X_X_0_1() {
    let pos = setup_game("k1b5/2p2b2/4p3/8/1pK2pb1/8/8/2p5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4d3", "c4b4", "c4d4", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2p2q2_4p3_8_1pK2pq1_8_8_2p5_w_X_X_0_1() {
    let pos = setup_game("k1q5/2p2q2/4p3/8/1pK2pq1/8/8/2p5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4d3", "c4b4", "c4d4", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2p2b2_4p3_8_1pK2pq1_8_8_2p5_w_X_X_0_1() {
    let pos = setup_game("k1r5/2p2b2/4p3/8/1pK2pq1/8/8/2p5 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4d3", "c4b4", "c4d4", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2P2r2_4P3_2PP4_2K1PPr1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r5/2P2r2/4P3/2PP4/2K1PPr1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "e4e5", "f4f5", "c5c6", "d5d6", "e6e7", "e6f7", "c4b3", "c4c3", "c4d3", "c4b4", "c4d4",
        "c4b5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2P2b2_4P3_2PP4_2K1PPb1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1b5/2P2b2/4P3/2PP4/2K1PPb1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "e4e5", "f4f5", "c5c6", "d5d6", "e6e7", "e6f7", "c4b3", "c4c3", "c4d3", "c4b4", "c4d4",
        "c4b5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2P2q2_4P3_2PP4_2K1PPq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1q5/2P2q2/4P3/2PP4/2K1PPq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "e4e5", "f4f5", "c5c6", "d5d6", "e6e7", "e6f7", "c4b3", "c4c3", "c4d3", "c4b4", "c4d4",
        "c4b5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2P2b2_4P3_2PP4_2K1PPq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r5/2P2b2/4P3/2PP4/2K1PPq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "e4e5", "f4f5", "c5c6", "d5d6", "e6e7", "e6f7", "c4b3", "c4c3", "c4d3", "c4b4", "c4d4",
        "c4b5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2p2r2_4p3_2pp4_2K1ppr1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r5/2p2r2/4p3/2pp4/2K1ppr1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4c3", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1b5_2p2b2_4p3_2pp4_2K1ppb1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1b5/2p2b2/4p3/2pp4/2K1ppb1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4c3", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1q5_2p2q2_4p3_2pp4_2K1ppq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1q5/2p2q2/4p3/2pp4/2K1ppq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4c3", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2p2b2_4p3_2pp4_2K1ppq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r5/2p2b2/4p3/2pp4/2K1ppq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = ["c4b3", "c4c3", "c4b5", "c4c5"];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r5_2p2b2_4p3_2PP4_2K1Ppq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r5/2p2b2/4p3/2PP4/2K1Ppq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "e4e5", "c5c6", "d5d6", "d5e6", "c4b3", "c4c3", "c4d3", "c4b4", "c4d4", "c4b5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k1r3b1_2P2P2_2p1p3_8_2K1pPq1_8_8_8_w_X_X_0_1() {
    let pos = setup_game("k1r3b1/2P2P2/2p1p3/8/2K1pPq1/8/8/8 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "f4f5", "f7f8q", "f7f8n", "f7f8r", "f7f8b", "f7g8q", "f7g8n", "f7g8r", "f7g8b", "c4b3",
        "c4c3", "c4b4", "c4d4", "c4c5",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k1r4_8_8_7b_b2Q4_5Q2_2Q5_3K4_w_X_X_0_1() {
    let pos = setup_game("1k1r4/8/8/7b/b2Q4/5Q2/2Q5/3K4 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "c2b3", "c2a4", "f3e2", "f3g4", "f3h5", "d4d2", "d4d3", "d4d5", "d4d6", "d4d7", "d4d8",
        "d1c1", "d1e1", "d1d2", "d1e2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_k6b_r7_8_Q7_3Q4_8_8_K4Q1r_w_X_X_0_1() {
    let pos = setup_game("k6b/r7/8/Q7/3Q4/8/8/K4Q1r w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "f1b1", "f1c1", "f1d1", "f1e1", "f1g1", "f1h1", "d4b2", "d4c3", "d4e5", "d4f6", "d4g7",
        "d4h8", "a5a2", "a5a3", "a5a4", "a5a6", "a5a7", "a1b1", "a1a2", "a1b2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k6_8_8_8_8_8_8_R3K2R_w_KQ_X_0_1() {
    let pos = setup_game("1k6/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1c1", "e1d1", "e1f1",
        "e1g1", "e1d2", "e1e2", "e1f2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k6_8_8_8_8_8_8_R3K2R_w_Kh_X_0_1() {
    let pos = setup_game("1k6/8/8/8/8/8/8/R3K2R w Kh - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1d1", "e1f1", "e1g1",
        "e1d2", "e1e2", "e1f2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k6_8_8_8_8_8_8_R3K2R_w_Qh_X_0_1() {
    let pos = setup_game("1k6/8/8/8/8/8/8/R3K2R w Qh - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1c1", "e1d1", "e1f1",
        "e1d2", "e1e2", "e1f2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k1r4_8_8_8_8_8_8_R3K2R_w_KQ_X_0_1() {
    let pos = setup_game("1k1r4/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1f1", "e1g1", "e1e2",
        "e1f2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k3r2_8_8_8_8_8_8_R3K2R_w_KQ_X_0_1() {
    let pos = setup_game("1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1c1", "e1d1", "e1d2",
        "e1e2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_1k1r1r2_8_8_8_8_8_8_R3K2R_w_KQ_X_0_1() {
    let pos = setup_game("1k1r1r2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a1b1", "a1c1", "a1d1", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "h1f1",
        "h1g1", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8", "e1e2",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_rnbqkbnr_1ppppppp_8_p7_8_N7_PPPPPPPP_R1BQKBNR_w_KQkq_X_2_1() {
    let pos = setup_game("rnbqkbnr/1ppppppp/8/p7/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 2 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4", "g2g3",
        "g2g4", "h2h3", "h2h4", "g1f3", "g1h3", "a3b1", "a3c4", "a3b5", "a1b1",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_rnbqkbnr_1ppppppp_8_p7_8_7N_PPPPPPPP_RNBQKB1R_w_KQkq_X_2_1() {
    let pos = setup_game("rnbqkbnr/1ppppppp/8/p7/8/7N/PPPPPPPP/RNBQKB1R w KQkq - 2 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4", "e2e3", "e2e4", "f2f3",
        "f2f4", "g2g3", "g2g4", "b1a3", "b1c3", "h3g1", "h3f4", "h3g5", "h1g1",
    ];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_8_8_8_8_8_r7_2k5_K7_w_X_X_0_1() {
    let pos = setup_game("8/8/8/8/8/r7/2k5/K7 w - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [];

    lists_match(&pos, moves, legal_moves);
}

#[test]
#[allow(non_snake_case)]
fn test_moves_from_8_8_8_8_8_1r6_2k5_K7_b_X_X_0_1() {
    let pos = setup_game("8/8/8/8/8/1r6/2k5/K7 b - - 0 1");
    let game = Game::new(pos.clone());
    let moves = game.get_legal_moves();
    let legal_moves = [
        "b3b1", "b3b2", "b3a3", "b3c3", "b3d3", "b3e3", "b3f3", "b3g3", "b3h3", "b3b4", "b3b5",
        "b3b6", "b3b7", "b3b8", "c2c1", "c2d1", "c2d2", "c2c3", "c2d3",
    ];

    lists_match(&pos, moves, legal_moves);
}
