use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::LlvmFunction,
};
use anyhow::Result;
use capstone::arch::arm::ArmInsn;
use inkwell::{IntPredicate, values::IntValue};
