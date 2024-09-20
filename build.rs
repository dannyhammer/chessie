use std::env;

fn main() {
    // Re-run the build script if it was changed
    println!("cargo::rerun-if-changed=build.rs");

    let outdir = env::var_os("OUT_DIR").unwrap();

    // Generate tables for RAY_BETWEEN[from][to]
    types::generate_ray_table_datfiles(&outdir).unwrap();

    // Generate attack .dat files for pieces
    types::generate_piece_attack_datfiles(&outdir).unwrap();

    // TODO: Generate magics
    // generate_magics(&outdir).unwrap();
}
