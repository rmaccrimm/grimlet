/// Convenience methods for constructing test programs. All functions will panic rather than return
/// an error if called incorrectly.
use capstone::{
    RegId,
    arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType},
};

use super::ArmInstruction;

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
        ..Default::default()
    }
}

pub fn op_reg_reg(opcode: ArmInsn, r1: u16, r2: u16, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2)],
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
        ..Default::default()
    }
}
