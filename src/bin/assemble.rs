use std::fs;
use std::process::{Command, Stdio};

use anyhow::{Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Instruction to disassemble
    instruction: String,

    /// Disassemble in THUMB mode
    #[arg(short, long, default_value_t = false)]
    thumb_mode: bool,
}

/// Assembles a single instruction and prints it as a hex value. Useful for creating test cases.
/// Relies on `gvasm` for assembling programs: https://github.com/velipso/gvasm
fn main() -> Result<()> {
    let args = Args::parse();

    let asm = format!(
        ".begin main\n\
        .{}\n\
        {}\n\
        .end",
        if args.thumb_mode { "thumb" } else { "arm" },
        args.instruction
    );
    let dir = "asm_tmp";
    let asm_path = format!("{}/a.gvasm", dir);
    if fs::exists(dir).expect("check for existing failed") {
        fs::remove_dir_all(dir).expect("delete existing failed");
    }
    fs::create_dir(dir)?;
    fs::write(&asm_path, asm)?;

    let run = || -> Result<String> {
        let status = Command::new("gvasm")
            .arg("make")
            .arg(&asm_path)
            .stderr(Stdio::inherit())
            .status()?;
        if !status.success() {
            bail!("assembly failed");
        }
        let bin = fs::read(format!("{}/a.gba", dir))?;
        if args.thumb_mode {
            Ok(format!(
                "{:#04x}",
                u16::from_le_bytes(bin[0..2].try_into()?)
            ))
        } else {
            Ok(format!(
                "{:#08x}",
                u32::from_le_bytes(bin[0..4].try_into()?)
            ))
        }
    };
    let result = run();
    fs::remove_dir_all(dir).expect("clean up failed");

    match result {
        Ok(s) => println!("{}", s),
        Err(e) => {
            println!("{}", e)
        }
    }
    Ok(())
}
