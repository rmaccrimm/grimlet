pub mod cons;

use std::collections::HashSet;
use std::fmt::Display;

use capstone::arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};

use crate::arm::cpu::{ArmMode, MainMemory, Reg};

// A single disassembled ARM instruction. Basically a clone of the Capstone instruction but
// easier to access since we know we're only working with ARM instructions.
#[derive(Clone, Debug)]
pub struct ArmInstruction {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: usize,
    pub repr: Option<String>,
    pub cond: ArmCC,
    pub mode: ArmMode,
    pub updates_flags: bool,
    pub regs_accessed: Vec<Reg>,
}

impl Default for ArmInstruction {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            cond: ArmCC::ARM_CC_AL,
            operands: Default::default(),
            addr: Default::default(),
            repr: None,
            mode: ArmMode::ARM,
            updates_flags: true,
            regs_accessed: vec![],
        }
    }
}

impl ArmInstruction {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn, mode: ArmMode) -> Self {
        let detail = cs
            .insn_detail(insn)
            .expect("failed to get instruction detail");

        let arch_detail = detail.arch_detail();
        let arm_detail = arch_detail.arm().expect("expected an arm instruction");

        let mut regs_accessed = Vec::new();
        let mut operands = Vec::new();

        for op in arch_detail.operands() {
            if let ArchOperand::ArmOperand(a) = op {
                if let ArmOperandType::Reg(reg_id) = a.op_type {
                    regs_accessed.push(Reg::from(reg_id))
                }
                operands.push(a);
            } else {
                panic!("not an ARM operand")
            }
        }

        let cond = arm_detail.cc();

        Self {
            opcode: ArmInsn::from(insn.id().0),
            operands: arch_detail
                .operands()
                .into_iter()
                .map(|a| match a {
                    ArchOperand::ArmOperand(op) => op,
                    _ => panic!("not an ARM operand"),
                })
                .collect(),
            addr: insn.address() as usize,
            repr: Some(insn.to_string()),
            cond,
            mode,
            updates_flags: arm_detail.update_flags(),
            regs_accessed,
        }
    }

    pub fn get_reg_op(&self, ind: usize) -> Reg {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{}\" missing operand {}", self, ind))
            .op_type
        {
            Reg::from(reg_id)
        } else {
            panic!("\"{}\" operand {} is not a register", self, ind);
        }
    }

    pub fn get_imm_op(&self, ind: usize) -> i32 {
        if let ArmOperandType::Imm(i) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{}\" missing operand {}", self, ind))
            .op_type
        {
            i
        } else {
            panic!("\"{}\" operand {} is not an immediate value", self, ind);
        }
    }
}

impl Display for ArmInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.repr {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "missing repr"),
        }
    }
}

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

pub trait Disasm {
    fn next_code_block(&self, mem: &MainMemory, addr: usize) -> CodeBlock;
}

pub struct MemoryDisassembler {
    cs: Capstone,
    current_mode: ArmMode,
}

impl Default for MemoryDisassembler {
    fn default() -> Self {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()
            .expect("failed to build capstone instance");
        Self {
            cs,
            current_mode: ArmMode::ARM,
        }
    }
}

impl Disasm for MemoryDisassembler {
    fn next_code_block(&self, mem: &MainMemory, start_addr: usize) -> CodeBlock {
        let instr_iter = mem.iter_word(start_addr).enumerate().map(move |(i, ch)| {
            let instructions = self
                .cs
                .disasm_count(ch, (start_addr as u64) + 4 * i as u64, 1)
                .expect("Capstone disassembly failed");

            let i = instructions
                .as_ref()
                .first()
                .expect("Capstone returned no instructions");
            ArmInstruction::from_cs_insn(&self.cs, i, self.current_mode)
        });
        CodeBlock::from_instructions(instr_iter, start_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disasm() {
        let bytes: [u8; 4] = [0b10010000, 0b00000001, 0b00010010, 0b11100000];
        let disasm = MemoryDisassembler::default();
        let res = &disasm.cs.disasm_all(&bytes, 0).unwrap()[0];
        println!(
            "{:#?}",
            ArmInstruction::from_cs_insn(&disasm.cs, res, ArmMode::ARM)
        );
    }
}
