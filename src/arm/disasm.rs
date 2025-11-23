use anyhow::Result;
use anyhow::anyhow;
use capstone::{
    Capstone, Insn,
    arch::{
        ArchOperand,
        arm::{ArmInsn, ArmOperand, ArmOperandType},
    },
};
use std::fmt::Display;

use crate::arm::cpu::Reg;

#[derive(Clone, Debug)]
pub struct ArmDisasm {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: u64,
    pub repr: String,
}

impl ArmDisasm {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn) -> Result<Self> {
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
            Err(anyhow!("Bad operand: not a register"))
        }
    }
}

impl Display for ArmDisasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}
