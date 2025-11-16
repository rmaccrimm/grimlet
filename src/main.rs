pub mod cpu;
pub mod jit;

use anyhow::{Result, anyhow};
use capstone::arch::arm::{ArmInsn, ArmOperand};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};
use inkwell::basic_block::InstructionIter;
use jit::Compiler;
use std::fmt::Display;

use cpu::ArmState;
use std::{env, fs};

use inkwell::context::Context;
use std::fs::File;
use std::io::Read;

/// Am I sticking with this name?
struct Grimlet<'a> {
    cs: Capstone,
    compiler: Compiler<'a>,
    state: ArmState,
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

struct CodeSegment {
    pub start_addr: u64,
    pub end_addr: u64,
    pub disasm_code: Vec<ArmDisasm>,
}

fn disasm_arm(cs: &Capstone, mem: &Vec<u8>, addr: u64) -> Result<Vec<ArmDisasm>> {
    let mut out = Vec::new();

    for (i, chunk) in mem.chunks(4).skip(addr as usize).enumerate() {
        let instructions = cs.disasm_count(chunk, addr + i as u64, 1)?;
        let i = instructions.as_ref().iter().next().unwrap();
        out.push(ArmDisasm::from_cs_insn(&cs, &i)?);
        let d = out.last().unwrap();
        if d.opcode == ArmInsn::ARM_INS_BX {
            break;
        }
    }
    Ok(out)
}

impl<'a> Grimlet<'a> {
    pub fn new(context: &'a Context, bios_path: &str) -> Result<Self> {
        let compiler = Compiler::new(&context);
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;

        let state = ArmState::with_bios(bios_path)?;

        Ok(Self {
            cs,
            compiler,
            state,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        let disasm = disasm_arm(&self.cs, &self.state.mem, 0)?;
        for d in disasm {
            println!("{}", d);
        }

        self.compiler.compile(0);
        let f = self.compiler.func_cache.get(&0).unwrap();
        unsafe {
            f.call(&mut self.state);
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
    grimlet.run();

    println!("{:?}", grimlet.state.regs);
    assert_eq!(grimlet.state.regs[0], 9);

    Ok(())
}
