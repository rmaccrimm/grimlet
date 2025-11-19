pub mod arm;
pub mod jit;

use crate::arm::cpu::ArmState;
use crate::jit::Compiler;
use anyhow::{Result, anyhow};
use capstone::arch::arm::{ArmInsn, ArmOperand};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};
use inkwell::basic_block::InstructionIter;
use inkwell::context::Context;
use std::fmt::Display;
use std::fs::File;
use std::io::Read;
use std::{env, fs};

/// Am I sticking with this name?
struct Grimlet<'a> {
    state: ArmState,
    disasm: Disassembler,
    compiler: Compiler<'a>,
}

#[derive(Clone, Debug)]
struct ArmDisasm {
    opcode: ArmInsn,
    operands: Vec<ArmOperand>,
    addr: u64,
    repr: String,
}

impl ArmDisasm {
    fn from_cs_insn(cs: &Capstone, insn: &Insn) -> Result<Self> {
        Ok(Self {
            opcode: ArmInsn::from(insn.id().0),
            operands: cs
                .insn_detail(&insn)?
                .arch_detail()
                .operands()
                .into_iter()
                .map(|a| match a {
                    ArchOperand::ArmOperand(op) => op,
                    _ => panic!("unexpected operand"),
                })
                .collect(),

            addr: insn.address(),
            repr: insn.to_string(),
        })
    }
}

impl Display for ArmDisasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}

enum ArmMode {
    ARM,
    THUMB,
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
        let comp = Compiler::new(&context);

        Ok(Self {
            state,
            disasm,
            compiler: comp,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        let entry_point = self.compiler.build_entry_point();
        loop {
            let curr_pc = self.state.pc();

            let func = match self.compiler.func_cache.get(&curr_pc) {
                Some(func) => func,
                None => {
                    let func = self.compiler.new_function(0);
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
                    self.compiler.compile(func).unwrap()
                }
            };
            self.compiler.dump();
            break;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // let bios_path = env::args().into_iter().skip(1).next().unwrap();
    let context = Context::create();
    let mut grimlet = Grimlet::new(&context, "gba_bios.bin")?;

    for i in 0..17 {
        grimlet.state.regs[i] = i as u32;
    }

    println!("{:?}", grimlet.state.regs);
    grimlet.run()?;

    println!("{:?}", grimlet.state.regs);
    assert_eq!(grimlet.state.pc(), 99);

    Ok(())
}
