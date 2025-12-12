use std::fs;
use std::process::{Command, Stdio};

use anyhow::{Result, anyhow, bail};
use capstone::Capstone;
use capstone::arch::BuildsCapstone as _;
use capstone::arch::arm::ArchMode;
use clap::Parser;
use grimlet::arm::cpu::ArmMode;
use grimlet::arm::disasm::ArmInstruction;

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    /// Instruction to disassemble
    instruction: String,

    /// Disassemble in THUMB mode
    #[arg(short, long, default_value_t = false)]
    thumb_mode: bool,
}

/// Command line utiltiy for inspecting the disassembler output for an instruction
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
    println!("{}", asm);
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
        let cs = Capstone::new()
            .arm()
            .mode(if args.thumb_mode {
                ArchMode::Thumb
            } else {
                ArchMode::Arm
            })
            .detail(true)
            .build()?;

        let bytes = if args.thumb_mode {
            &bin[0..2]
        } else {
            &bin[0..4]
        };
        let instrs = cs.disasm_count(bytes, 0, 1)?;
        let instr = instrs
            .as_ref()
            .first()
            .ok_or(anyhow!("disassembly failed"))?;

        Ok(format!(
            "{:#?}",
            ArmInstruction::from_cs_insn(
                &cs,
                instr,
                if args.thumb_mode {
                    ArmMode::THUMB
                } else {
                    ArmMode::ARM
                }
            )
        ))
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
