/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::env;

use chessie_types::{generate_piece_attack_datfiles, generate_ray_table_datfiles};

fn main() {
    // Re-run the build script if it was changed
    println!("cargo:rerun-if-changed=build.rs");

    let outdir = env::var_os("OUT_DIR").unwrap();

    // Generate tables for RAY_BETWEEN[from][to]
    generate_ray_table_datfiles(&outdir).unwrap();

    // Generate attack .dat files for pieces
    generate_piece_attack_datfiles(&outdir).unwrap();

    // TODO: Generate magics
    // generate_magics(&outdir).unwrap();
}
