use std::env;
use std::mem::size_of;

use anyhow::Result;
use grimlet::arm::cpu::ArmState;
use grimlet::arm::disasm::{ArmInstruction, MemoryDisassembler};
use grimlet::emulator::Emulator;
use inkwell::context::Context;

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let context = Context::create();
    let disasm = MemoryDisassembler::default();
    let mut emulator = Emulator::new(&context, disasm, Some(&bios_path))?;
    // run indefinitely
    let exit: Option<fn(&ArmState) -> bool> = None;
    emulator.run(exit);
    println!("{}", size_of::<ArmInstruction>());
    Ok(())
}
