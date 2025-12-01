#![allow(dead_code)]

pub mod arm;
pub mod jit;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::{ArmDisasm, Disassembler};
use crate::jit::{Compiler, EntryPoint, FunctionCache};
use anyhow::Result;
use inkwell::context::Context;
use std::env;

/// Am I sticking with this name?
struct Grimlet<'ctx> {
    state: ArmState,
    disasm: Disassembler,
    compiler: Compiler<'ctx>,
    entry_point: EntryPoint<'ctx>,
    func_cache: FunctionCache<'ctx>,
}

impl<'ctx> Grimlet<'ctx> {
    pub fn new(context: &'ctx Context, bios_path: &str) -> Result<Self> {
        let state = ArmState::with_bios(bios_path)?;
        let disasm = Disassembler::new()?;
        let mut compiler = Compiler::new(context)?;
        let entry_point = compiler.compile_entry_point()?;
        let func_cache = FunctionCache::new();

        Ok(Self {
            state,
            disasm,
            compiler,
            entry_point,
            func_cache,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        let mut curr_pc = 0;

        loop {
            let code_block = self.disasm.next_code_block(&self.state.mem, curr_pc);
            println!("{}", code_block);
            curr_pc = code_block.instrs.last().unwrap().addr + 4;
        }
    }
}

fn main() -> Result<()> {
    let bios_path = env::args().nth(1).unwrap();
    let context = Context::create();
    let mut emulator = Grimlet::new(&context, &bios_path)?;
    emulator.run()?;
    println!("{}", size_of::<ArmDisasm>());
    Ok(())
}
