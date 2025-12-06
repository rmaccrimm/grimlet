#![allow(dead_code)]

pub mod arm;
pub mod emulator;
pub mod jit;

use std::env;

use anyhow::Result;
use inkwell::context::Context;

use crate::arm::disasm::ArmInstruction;
use crate::emulator::Emulator;

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let context = Context::create();
    let mut emulator = Emulator::new(&context, &bios_path)?;
    emulator.run();
    println!("{}", size_of::<ArmInstruction>());
    Ok(())
}
