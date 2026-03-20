use std::fmt::Debug;

use anyhow::Result;
use clap::Parser;
use grimlet::arm::disasm::Disassembler;
use grimlet::arm::state::ArmState;
use grimlet::emulator::{DebugOutput, Emulator};

#[derive(Parser, Debug)]
#[command(version, about, long_about=None)]
struct Args {
    bios_path: String,

    #[arg(value_enum)]
    debug_output: Option<DebugOutput>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let disasm = Disassembler::default();
    let mut emulator = Emulator::new(disasm);
    emulator.load_rom(args.bios_path, 0)?;
    let exit = |_: &ArmState| -> bool { false };
    emulator.run(exit, args.debug_output);
    Ok(())
}
