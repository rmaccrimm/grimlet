/// Convenience methods for constructing test programs. All functions will panic rather than return
/// an error if called incorrectly.
use capstone::{
    RegId,
    arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType, ArmReg},
};

use super::ArmInstruction;
use crate::arm::cpu::REG_ITEMS;

fn reg(r: usize) -> ArmOperand {
    // Janky, inverse of the conversion done in cpu::Reg. Would like to eliminate this
    let reg_id = match r {
        0 => ArmReg::ARM_REG_R0,
        1 => ArmReg::ARM_REG_R1,
        2 => ArmReg::ARM_REG_R2,
        3 => ArmReg::ARM_REG_R3,
        4 => ArmReg::ARM_REG_R4,
        5 => ArmReg::ARM_REG_R5,
        6 => ArmReg::ARM_REG_R6,
        7 => ArmReg::ARM_REG_R7,
        8 => ArmReg::ARM_REG_R8,
        9 => ArmReg::ARM_REG_R9,
        10 => ArmReg::ARM_REG_R10,
        11 => ArmReg::ARM_REG_R11,
        12 => ArmReg::ARM_REG_R12,
        13 => ArmReg::ARM_REG_SP,
        14 => ArmReg::ARM_REG_LR,
        15 => ArmReg::ARM_REG_PC,
        16 => ArmReg::ARM_REG_CPSR,
        _ => panic!("unhandled reg"),
    };
    ArmOperand {
        op_type: ArmOperandType::Reg(RegId(reg_id as u16)),
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

pub fn op_reg_imm(opcode: ArmInsn, r: usize, i: i32, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r), imm(i)],
        regs_accessed: vec![REG_ITEMS[r]],
        ..Default::default()
    }
}

pub fn op_reg_reg(opcode: ArmInsn, r1: usize, r2: usize, cc: Option<ArmCC>) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2)],
        regs_accessed: vec![REG_ITEMS[r1], REG_ITEMS[r2]],
        ..Default::default()
    }
}

pub fn op_reg_reg_imm(
    opcode: ArmInsn,
    r1: usize,
    r2: usize,
    i: i32,
    cc: Option<ArmCC>,
) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2), imm(i)],
        regs_accessed: vec![REG_ITEMS[r1], REG_ITEMS[r2]],
        ..Default::default()
    }
}

pub fn op_reg_reg_reg(
    opcode: ArmInsn,
    r1: usize,
    r2: usize,
    r3: usize,
    cc: Option<ArmCC>,
) -> ArmInstruction {
    ArmInstruction {
        opcode,
        cond: cc.unwrap_or(ArmCC::ARM_CC_AL),
        operands: vec![reg(r1), reg(r2), reg(r3)],
        regs_accessed: vec![REG_ITEMS[r1], REG_ITEMS[r2], REG_ITEMS[r3]],
        ..Default::default()
    }
}
