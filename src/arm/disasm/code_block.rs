use std::collections::HashSet;
use std::fmt::Display;

use capstone::arch::arm::ArmInsn;

use crate::arm::cpu::Reg;
use crate::arm::disasm::instruction::ArmInstruction;

/// A single batch of instrutions to be compiled into an LLVM function
#[derive(Debug)]
pub struct CodeBlock {
    pub instrs: Vec<ArmInstruction>,
    pub regs_accessed: HashSet<Reg>,
    pub start_addr: usize,
}

impl Display for CodeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "---------------")?;
        for instr in self.instrs.iter() {
            write!(f, "0x{:02x}:  ", instr.addr)?;
            match instr.repr {
                Some(_) => writeln!(f, "{}", instr)?,
                None => writeln!(f, "{:?}", instr)?,
            }
        }
        writeln!(f, "regs accessed: {:?}", self.regs_accessed)?;
        writeln!(f, "---------------")
    }
}

impl CodeBlock {
    pub fn from_instructions(
        instr_iter: impl Iterator<Item = ArmInstruction>,
        start_addr: usize,
    ) -> Self {
        let mut instrs = Vec::new();
        let mut regs_read = HashSet::new();
        for instr in instr_iter {
            instrs.push(instr);
            let instr = instrs.last().unwrap();
            match instr.opcode {
                ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => {
                    break;
                }
                _ => {}
            }
            for r in instr.regs_accessed.iter() {
                regs_read.insert(*r);
            }
        }
        CodeBlock {
            instrs,
            start_addr,
            regs_accessed: regs_read,
        }
    }
}
