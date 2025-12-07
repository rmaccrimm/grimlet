/// Convenience methods for constructing test programs. All functions will panic rather than return
/// an error if called incorrectly.
use capstone::{
    RegId,
    arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType},
};

use super::ArmInstruction;
use crate::arm::cpu::Reg;

fn reg(r: u16) -> ArmOperand {
    ArmOperand {
        op_type: ArmOperandType::Reg(RegId(r)),
        ..Default::default()
    }
}

fn imm(i: i32) -> ArmOperand {
    ArmOperand {
        op_type: ArmOperandType::Imm(i),
        ..Default::default()
    }
}

pub fn op_imm(opcode: ArmInsn, i: i32, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![imm(i)],
        ..Default::default()
    }
}

pub fn op_reg_imm(opcode: ArmInsn, r: u16, i: i32, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r), imm(i)],
        regs_read: vec![Reg::from(r as usize)],
        ..Default::default()
    }
}

pub fn op_reg_reg(opcode: ArmInsn, r1: u16, r2: u16, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2)],
        regs_read: vec![Reg::from(r1 as usize), Reg::from(r2 as usize)],
        ..Default::default()
    }
}

pub fn op_reg_reg_imm(
    opcode: ArmInsn,
    r1: u16,
    r2: u16,
    i: i32,
    cc: Option<ArmCC>,
) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2), imm(i)],
        regs_read: vec![Reg::from(r1 as usize), Reg::from(r2 as usize)],
        ..Default::default()
    }
}

pub fn op_reg_reg_reg(
    opcode: ArmInsn,
    r1: u16,
    r2: u16,
    r3: u16,
    cc: Option<ArmCC>,
) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2), reg(r3)],
        regs_read: vec![
            Reg::from(r1 as usize),
            Reg::from(r2 as usize),
            Reg::from(r3 as usize),
        ],
        ..Default::default()
    }
}
