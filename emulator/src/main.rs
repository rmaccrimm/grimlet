pub mod arm;
pub mod emulator;
pub mod jit;

use std::env;
use std::mem::size_of;

use anyhow::Result;
use inkwell::context::Context;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::{ArmInstruction, MemoryDisassembler};
use crate::emulator::Emulator;

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
