pub mod cons;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use capstone::RegId;
use capstone::arch::BuildsCapstone;
use capstone::arch::arm::ArmCC;
use capstone::{
    Capstone, Insn,
    arch::{
        ArchOperand,
        arm::{ArmInsn, ArmOperand, ArmOperandType},
    },
};
use std::fmt::Display;

use crate::arm::cpu::ArmMode;
use crate::arm::cpu::MainMemory;
use crate::arm::cpu::Reg;

// A single disassembled ARM instruction
#[derive(Clone, Debug)]
pub struct ArmDisasm {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: usize,
    pub repr: String,
    pub cond: ArmCC,
}

impl Default for ArmDisasm {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            cond: ArmCC::ARM_CC_AL,
            operands: Default::default(),
            addr: Default::default(),
            repr: Default::default(),
        }
    }
}

impl ArmDisasm {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn) -> Self {
        let detail = cs
            .insn_detail(insn)
            .expect("failed to get instruction detail");
        let arch_detail = detail.arch_detail();
        let cond = arch_detail.arm().unwrap().cc();
        Self {
            opcode: ArmInsn::from(insn.id().0),
            operands: arch_detail
                .operands()
                .into_iter()
                .map(|a| match a {
                    ArchOperand::ArmOperand(op) => op,
                    _ => panic!("unexpected operand"),
                })
                .collect(),
            addr: insn.address() as usize,
            repr: insn.to_string(),
            cond,
        }
    }

    pub fn get_reg_op(&self, ind: usize) -> Result<Reg> {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .ok_or(anyhow!("missing operand {}", ind))?
            .op_type
        {
            Ok(Reg::from(reg_id.0 as usize))
        } else {
            Err(anyhow!("Bad operand: not a register"))
        }
    }

    pub fn get_imm_op(&self, ind: usize) -> Result<i32> {
        if let ArmOperandType::Imm(i) = self
            .operands
            .get(ind)
            .ok_or(anyhow!("missing operand {}", ind))?
            .op_type
        {
            Ok(i)
        } else {
            Err(anyhow!("Bad operand: not an immediate value"))
        }
    }
}

impl Display for ArmDisasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}

#[derive(Debug)]
pub struct CodeBlock {
    pub instrs: Vec<ArmDisasm>,
    pub start_addr: usize,
    pub labels: Vec<usize>,
}

impl Display for CodeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut lbl_iter = self.labels.iter().enumerate();
        let mut lbl_addr = lbl_iter.next();
        for instr in self.instrs.iter() {
            if let Some((i, addr)) = lbl_addr
                && addr == &instr.addr
            {
                writeln!(f, "{}:", i)?;
                lbl_addr = lbl_iter.next();
            }
            writeln!(f, "\t{}", instr)?;
        }
        Ok(())
    }
}

impl CodeBlock {
    pub fn from_instructions(
        instr_iter: impl Iterator<Item = ArmDisasm>,
        start_addr: usize,
    ) -> Self {
        let mut instrs = Vec::new();
        let mut labels = Vec::new();

        for instr in instr_iter {
            match instr.opcode {
                ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => {
                    // TODO can these be negative? Pretty sure it's translated to abs address
                    if let Ok(target) = instr.get_imm_op(0) {
                        let t = target as usize;
                        if t < instr.addr && t >= start_addr {
                            // This is a loop, need a label at the target address
                            labels.push(instr.addr);
                            instrs.push(instr);
                        } else {
                            instrs.push(instr);
                            break;
                        }
                    }
                }
                _ => {
                    instrs.push(instr);
                }
            }
        }
        CodeBlock {
            instrs,
            start_addr,
            labels,
        }
    }
}

pub struct Disassembler {
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

    pub fn next_code_block(&self, mem: &MainMemory, start_addr: usize) -> CodeBlock {
        let instr_iter = mem.iter_word(start_addr).enumerate().map(move |(i, ch)| {
            let instructions = self
                .cs
                .disasm_count(ch, (start_addr as u64) + 4 * i as u64, 1)
                .expect("Capstone disassembly failed");

            let i = instructions
                .as_ref()
                .first()
                .expect("Capstone returned no instructions");
            ArmDisasm::from_cs_insn(&self.cs, i)
        });
        CodeBlock::from_instructions(instr_iter, start_addr)
    }
}

#[cfg(test)]
mod tests {}
