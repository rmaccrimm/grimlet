use std::env;

use anyhow::Result;
use grimlet::arm::cpu::ArmState;
use grimlet::arm::disasm::MemoryDisassembler;
use grimlet::emulator::Emulator;

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let disasm = MemoryDisassembler::default();
    let mut emulator = Emulator::new(disasm, Some(&bios_path))?;
    let exit = |_: &ArmState| -> bool { false };
    emulator.run(exit);
    Ok(())
}
