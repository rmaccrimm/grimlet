#![allow(dead_code)]

pub mod arm;
pub mod jit;

use crate::arm::cpu::{ArmMode, ArmState};
use crate::arm::disasm::ArmDisasm;
use crate::jit::Compiler;
use anyhow::Result;
use capstone::arch::arm::{ArmInsn, ArmOperand};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};

use inkwell::context::Context;
use std::env;
use std::fmt::Display;

/// Am I sticking with this name?
struct Grimlet<'a> {
    state: ArmState,
    disasm: Disassembler,
    compiler: Compiler<'a>,
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
        mem: &Vec<u8>,
        start_addr: u64,
        _mode: ArmMode,
    ) -> impl Iterator<Item = ArmDisasm> {
        mem.chunks(4)
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

impl<'a> Grimlet<'a> {
    pub fn new(context: &'a Context, bios_path: &str) -> Result<Self> {
        let state = ArmState::with_bios(bios_path)?;
        let disasm = Disassembler::new()?;
        let compiler = Compiler::new(&context)?;

        Ok(Self {
            state,
            disasm,
            compiler,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        loop {
            let curr_pc = self.state.pc();
            let func_key = match self.compiler.lookup_function(curr_pc) {
                Some(func) => func,
                None => {
                    let func = self.compiler.new_function(0)?;
                    for insn in
                        self.disasm
                            .iter_insns(&self.state.mem, curr_pc as u64, ArmMode::ARM)
                    {
                        println!("{}", insn);
                        // let should_exit = func.append_insn(insn);
                        let should_exit = true;
                        self.compiler.append_insn(&func, 0);

                        if should_exit {
                            break;
                        }
                    }
                    self.compiler.compile(func)?
                }
            };
            self.compiler.call_function(func_key, &mut self.state)?;
            self.compiler.dump();
            break;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let bios_path = env::args().into_iter().skip(1).next().unwrap();
    let context = Context::create();
    let mut grimlet = Grimlet::new(&context, &bios_path)?;

    for i in 0..17 {
        grimlet.state.regs[i] = i as u32;
    }

    println!("{:?}", grimlet.state.regs);
    grimlet.run()?;

    println!("{:?}", grimlet.state.regs);
    assert_eq!(grimlet.state.pc(), 99);

    Ok(())
}
