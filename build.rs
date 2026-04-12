use std::error::Error;
use std::fs;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=tests/asm");
    for f in fs::read_dir("tests/asm")? {
        let asm_path = f?.path();
        let Some(stem) = asm_path.file_stem() else {
            continue;
        };
        let Some(ext) = asm_path.extension() else {
            continue;
        };
        if ext != "gvasm" {
            continue;
        }
        if !fs::exists("tests/bin")? {
            fs::create_dir("tests/bin")?;
        }
        let status = Command::new("gvasm")
            .arg("make")
            .arg(&asm_path)
            .arg("-o")
            .arg(format!(
                "tests/bin/{}.gba",
                stem.to_str().expect("invalid file name")
            ))
            .stderr(Stdio::inherit())
            .status()?;
        if !status.success() {
            panic!("gvasm failed");
        }
    }
    Ok(())
}
