use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use chessie::*;

fn perft(game: &Game, depth: usize) -> u64 {
    if depth == 0 {
        return 1;
    }

    // Recursively accumulate the nodes from the remaining depths
    game.get_legal_moves().into_iter().fold(0, |nodes, mv| {
        nodes + perft(&game.with_move_made(mv), depth - 1)
    })
}

fn perft_benchmark(c: &mut Criterion) {
    let kiwipete = Game::from_fen(FEN_KIWIPETE).unwrap();
    c.bench_function("Kiwipete Perft 4", |b| {
        b.iter(|| {
            let kiwipete = black_box(&kiwipete);
            let depth = black_box(4);
            black_box(perft(kiwipete, depth))
        });
    });
}

// criterion_group!(benches, perft_benchmark);
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(Duration::from_secs(60));
    targets = perft_benchmark
}
criterion_main!(benches);
