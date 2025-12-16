use std::env;

use anyhow::Result;
use grimlet::arm::disasm::Disassembler;
use grimlet::arm::state::ArmState;
use grimlet::emulator::Emulator;

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let disasm = Disassembler::default();
    let mut emulator = Emulator::new(disasm, Some(&bios_path))?;
    let exit = |_: &ArmState| -> bool { false };
    emulator.run(exit, None);
    Ok(())
}
