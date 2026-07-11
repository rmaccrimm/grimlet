use std::env;
extern crate cbindgen;

fn main() {
    // println!("cargo::rerun-if-changed=src/lib.rs");
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = format!("{}/../macos/Grimlet/Grimlet/libbindings.h", crate_dir,);

    cbindgen::generate(crate_dir)
        .expect("failed to generate bindings")
        .write_to_file(output_file);
}
