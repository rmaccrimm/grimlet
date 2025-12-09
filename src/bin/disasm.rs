use std::process::{Command, Stdio};
use std::{env, fs};

use anyhow::{Result, anyhow, bail};
use capstone::Capstone;
use capstone::arch::BuildsCapstone as _;
use capstone::arch::arm::ArchMode;
use grimlet::arm::cpu::ArmMode;
use grimlet::arm::disasm::ArmInstruction;

/// Command line utiltiy for inspecting the disassembler output for an instruction
/// Relies on `gvasm` for assembling programs: https://github.com/velipso/gvasm
fn main() -> Result<()> {
    let cmd = env::args().nth(1).unwrap();
    let asm = format!(
        ".begin main\n\
        .arm\n\
        {}\n\
        .end",
        cmd
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
        let cs = Capstone::new()
            .arm()
            .mode(ArchMode::Arm)
            .detail(true)
            .build()?;

        let instrs = cs.disasm_count(&bin[0..4], 0, 1)?;
        let instr = instrs
            .as_ref()
            .first()
            .ok_or(anyhow!("disassembly failed"))?;

        Ok(format!(
            "{:#?}",
            ArmInstruction::from_cs_insn(&cs, instr, ArmMode::ARM)
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
