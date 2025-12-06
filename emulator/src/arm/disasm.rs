pub mod cons;

use capstone::arch::arm::ArmCC;
use capstone::arch::{self, BuildsCapstone};
use capstone::{
    Capstone, Insn,
    arch::{
        ArchOperand,
        arm::{ArmInsn, ArmOperand, ArmOperandType},
    },
};
use std::fmt::Display;

use crate::arm::cpu::Reg;
use crate::arm::cpu::{ArmMode, MainMemory};

// A single disassembled ARM instruction. Basically a clone of the Capstone instruction but
// easier to access since we know we're only working with ARM instructions.
#[derive(Clone, Debug)]
pub struct ArmDisasm {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: usize,
    pub repr: Option<String>,
    pub cond: ArmCC,
    pub mode: ArmMode,
    pub updates_flags: bool,
}

impl Default for ArmDisasm {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            cond: ArmCC::ARM_CC_AL,
            operands: Default::default(),
            addr: Default::default(),
            repr: None,
            mode: ArmMode::ARM,
            updates_flags: true,
        }
    }
}

impl ArmDisasm {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn, mode: ArmMode) -> Self {
        let detail = cs
            .insn_detail(insn)
            .expect("failed to get instruction detail");
        let arch_detail = detail.arch_detail();
        let arm_detail = arch_detail.arm().expect("expected an arm instruction");
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
        }
    }

    pub fn get_reg_op(&self, ind: usize) -> Reg {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{}\" missing operand {}", self, ind))
            .op_type
        {
            Reg::from(reg_id.0 as usize)
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

impl Display for ArmDisasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.repr {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "missing repr"),
        }
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
            match instr.repr {
                Some(_) => writeln!(f, "\t{}", instr)?,
                None => writeln!(f, "\t{:?}", instr)?,
            }
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
        let labels = Vec::new();

        for instr in instr_iter {
            match instr.opcode {
                ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => {
                    // TODO can these be negative? Pretty sure it's translated to abs address
                    // let target = instr.get_imm_op(0) as usize;
                    instrs.push(instr);
                    break;
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
    current_mode: ArmMode,
}

impl Default for Disassembler {
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

impl Disassembler {
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
            ArmDisasm::from_cs_insn(&self.cs, i, self.current_mode)
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
        let disasm = Disassembler::default();
        let res = &disasm.cs.disasm_all(&bytes, 0).unwrap()[0];
        println!(
            "{:#?}",
            ArmDisasm::from_cs_insn(&disasm.cs, res, ArmMode::ARM)
        );
    }
}
