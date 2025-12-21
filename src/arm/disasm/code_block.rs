use std::collections::HashSet;
use std::fmt::Display;

use capstone::arch::arm::{ArmInsn, ArmOperandType};

use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::Reg;

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
        let mut regs_accessed = HashSet::new();

        for instr in instr_iter {
            instrs.push(instr);
            let instr = instrs.last().unwrap();
            for a in instr.operands.iter() {
                match a.op_type {
                    ArmOperandType::Reg(reg_id) => {
                        regs_accessed.insert(Reg::from(reg_id));
                    }
                    ArmOperandType::Mem(arm_op_mem) => {
                        let base = arm_op_mem.base();
                        let index = arm_op_mem.index();
                        if base.0 != 0 {
                            regs_accessed.insert(Reg::from(base));
                        }
                        if index.0 != 0 {
                            regs_accessed.insert(Reg::from(index));
                        }
                    }
                    _ => (),
                }
            }
            match instr.opcode {
                ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => {
                    break;
                }
                _ => {}
            }
        }
        CodeBlock {
            instrs,
            start_addr,
            regs_accessed,
        }
    }
}
