#![allow(dead_code)]

pub mod arm;
pub mod jit;

use crate::arm::cpu::{ArmMode, ArmState, MainMemory};
use crate::arm::disasm::ArmDisasm;
use crate::jit::{Compiler, EntryPoint, FunctionCache};
use anyhow::Result;
use capstone::arch::arm::{ArmInsn, ArmOperand};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};

use inkwell::context::Context;
use std::env;
use std::fmt::Display;

/// Am I sticking with this name?
struct Grimlet<'ctx> {
    state: ArmState,
    disasm: Disassembler,
    compiler: Compiler<'ctx>,
    entry_point: EntryPoint<'ctx>,
    func_cache: FunctionCache<'ctx>,
}

struct Disassembler {
    cs: Capstone,
}

impl Disassembler {
    pub fn new() -> Result<Self> {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;
        Ok(Self { cs })
    }

    pub fn iter_insns(
        &self,
        mem: &MainMemory,
        start_addr: u64,
        _mode: ArmMode,
    ) -> impl Iterator<Item = ArmDisasm> {
        mem.bios
            .chunks(4)
            .skip(start_addr as usize)
            .enumerate()
            .map(move |(i, ch)| {
                let instructions = self
                    .cs
                    .disasm_count(ch, start_addr + 4 * i as u64, 1)
                    .unwrap();
                let i = instructions.as_ref().iter().next().unwrap();
                ArmDisasm::from_cs_insn(&self.cs, &i).unwrap()
            })
    }
}

impl<'ctx> Grimlet<'ctx> {
    pub fn new(context: &'ctx Context, bios_path: &str) -> Result<Self> {
        let state = ArmState::with_bios(bios_path)?;
        let disasm = Disassembler::new()?;
        let mut compiler = Compiler::new(&context)?;
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
        loop {
            let curr_pc = self.state.pc() as u64;
            let func = match self.func_cache.get(&curr_pc) {
                Some(func) => func,
                None => {
                    let func = self.compiler.new_function(curr_pc, &self.func_cache)?;
                    for insn in
                        self.disasm
                            .iter_insns(&self.state.mem, curr_pc as u64, ArmMode::ARM)
                    {
                        println!("{}", insn);
                        break;
                    }
                    let compiled = func.compile()?;
                    self.func_cache.insert(curr_pc, compiled);
                    self.func_cache.get(&curr_pc).unwrap()
                }
            };
            unsafe { self.entry_point.call(&mut self.state, func.as_raw()) };
            break;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let bios_path = env::args().into_iter().skip(1).next().unwrap();
    let context = Context::create();
    let mut grimlet = Grimlet::new(&context, &bios_path)?;
    grimlet.run()?;
    Ok(())
}
