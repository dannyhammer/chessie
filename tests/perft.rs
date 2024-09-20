use chessie::{perft, Game};

fn test_perft_fen_nodes(depth: usize, fen: &str, expected: u64) {
    let position = Game::from_fen(fen).unwrap();
    let res = perft(&position, depth);
    assert_eq!(res, expected);
}

#[test]
fn test_standard_epd() {
    let contents = std::fs::read_to_string("tests/standard.epd").unwrap();

    for (i, entry) in contents.lines().enumerate() {
        let mut parts = entry.split(';');

        let fen = parts.next().unwrap().trim();

        for perft_data in parts {
            let depth = usize::from_str_radix(perft_data.get(1..2).unwrap().trim(), 10).unwrap();
            let expected = u64::from_str_radix(perft_data.get(3..).unwrap().trim(), 10).unwrap();
            // println!("perft({depth}, \"{fen}\") := {expected}");

            let mut position = Game::from_fen(fen).unwrap();

            let nodes = perft(&mut position, depth);

            assert_eq!(
                nodes, expected,
                "\n{i} Perft({depth}, \"{fen}\") failed\nExpected: {expected}\nGot: {nodes}"
            );
        }
    }
}

#[cfg(test)]
mod kiwipete_perft {

    use super::*;

    #[test]
    fn kiwipete_perft_1() {
        test_perft_fen_nodes(
            1,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 ",
            48,
        );
    }

    #[test]
    fn kiwipete_perft_2() {
        test_perft_fen_nodes(
            2,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            2039,
        );
    }

    #[test]
    fn kiwipete_perft_3() {
        test_perft_fen_nodes(
            3,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            97862,
        );
    }

    #[test]
    fn kiwipete_perft_4() {
        test_perft_fen_nodes(
            4,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            4085603,
        );
    }
    #[test]
    fn kiwipete_perft_5() {
        test_perft_fen_nodes(
            5,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 ",
            193690690,
        );
    }

    /*
    #[test]
    fn kiwipete_perft_6() {
        test_perft_fen_nodes(
            6,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 ",
            8031647685,
        );
    }
     */
}

#[cfg(test)]
mod promotion_perft {
    use crate::test_perft_fen_nodes;

    #[test]
    fn test_promotion_perft_1() {
        test_perft_fen_nodes(1, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 24);
    }
    #[test]
    fn test_promotion_perft_2() {
        test_perft_fen_nodes(2, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 496);
    }
    #[test]
    fn test_promotion_perft_3() {
        test_perft_fen_nodes(3, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 9483);
    }
    #[test]
    fn test_promotion_perft_4() {
        test_perft_fen_nodes(4, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 182838);
    }
    #[test]
    fn test_promotion_perft_5() {
        test_perft_fen_nodes(5, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 3605103);
    }

    #[test]
    fn test_promotion_perft_6() {
        test_perft_fen_nodes(6, "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1 ", 71179139);
    }
}

/*
#[test]
fn perft_0() {
    test_starting_pos_perft(0);
}

#[test]
fn perft_1() {
    test_starting_pos_perft(1);
}

#[test]
fn perft_2() {
    test_starting_pos_perft(2);
}

#[test]
fn perft_3() {
    test_starting_pos_perft(3);
}

#[test]
fn perft_4() {
    test_starting_pos_perft(4);
}

#[test]
fn perft_5() {
    test_starting_pos_perft(5);
}

 */

/// https://www.chessprogramming.net/perfect-perft/
#[cfg(test)]
mod simple_perfts {
    use super::*;

    #[test]
    fn test_simple_perft_1() {
        test_perft_fen_nodes(
            6,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            119060324,
        );
    }

    #[test]
    fn test_simple_perft_2() {
        test_perft_fen_nodes(
            5,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            193690690,
        );
    }

    // #[test]
    // fn test_simple_perft_3() {
    //     test_perft_fen_nodes(7, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 178633661);
    // }

    // #[test]
    // fn test_simple_perft_4() {
    //     test_perft_fen_nodes(
    //         6,
    //         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    //         706045033,
    //     );
    // }

    #[test]
    fn test_simple_perft_5() {
        test_perft_fen_nodes(5, "1k6/1b6/8/8/7R/8/8/4K2R b K - 0 1", 1063513);
    }
}

/// https://www.chessprogramming.net/perfect-perft/
#[cfg(test)]
mod special_perfts {
    use super::*;

    #[test]
    fn test_special_perft_illegal_ep_move_1() {
        test_perft_fen_nodes(6, "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 1134888);
    }

    #[test]
    fn test_special_perft_illegal_ep_move_2() {
        test_perft_fen_nodes(6, "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 1015133);
    }

    #[test]
    fn test_special_perft_ep_capture_checks_opponent() {
        test_perft_fen_nodes(6, "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 1440467);
    }

    #[test]
    fn test_special_perft_short_castling_gives_check() {
        test_perft_fen_nodes(6, "5k2/8/8/8/8/8/8/4K2R w K - 0 1", 661072);
    }

    #[test]
    fn test_special_perft_long_castling_gives_check() {
        test_perft_fen_nodes(6, "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 803711);
    }

    #[test]
    fn test_special_perft_castling_rights() {
        test_perft_fen_nodes(4, "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 1274206);
    }

    #[test]
    fn test_special_perft_castling_prevented() {
        test_perft_fen_nodes(4, "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 1720476);
    }

    #[test]
    fn test_special_perft_promote_out_of_check() {
        test_perft_fen_nodes(6, "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 3821001);
    }

    #[test]
    fn test_special_perft_discovered_check() {
        test_perft_fen_nodes(5, "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 1004658);
    }

    #[test]
    fn test_special_perft_promote_to_give_check() {
        test_perft_fen_nodes(6, "4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 217342);
    }

    #[test]
    fn test_special_perft_under_promote_to_give_check() {
        test_perft_fen_nodes(6, "8/P1k5/K7/8/8/8/8/8 w - - 0 1", 92683);
    }

    #[test]
    fn test_special_perft_self_stalemate() {
        test_perft_fen_nodes(6, "K1k5/8/P7/8/8/8/8/8 w - - 0 1", 2217);
    }

    #[test]
    fn test_special_perft_stalemate_and_checkmate_1() {
        test_perft_fen_nodes(7, "8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 567584);
    }

    #[test]
    fn test_special_perft_stalemate_and_checkmate_2() {
        test_perft_fen_nodes(4, "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 23527);
    }
}
