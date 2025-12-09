use std::env;
use std::mem::size_of;

use anyhow::Result;
use grimlet::arm::cpu::ArmState;
use grimlet::arm::disasm::{ArmInstruction, Disasm, MemoryDisassembler};
use grimlet::emulator::Emulator;
use inkwell::context::Context;

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let state = ArmState::with_bios(bios_path)?;

    let disasm = MemoryDisassembler::default();
    for instr in disasm.next_code_block(&state.mem, 0).instrs {
        println!("{:#?}", instr);
    }
    Ok(())
}
