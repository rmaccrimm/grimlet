use inkwell::values::PointerValue;

use crate::{
    arm::{cpu::ArmState, disasm::ArmDisasm},
    jit::{Compiler, LlvmFunction},
};

impl<'ctx> Compiler<'ctx> {
    pub fn arm_b(&self, instr: &ArmDisasm) {
        let bd = &self.builder;
    }

    pub fn arm_bl(&self, instr: &ArmDisasm) {
        todo!();
    }

    pub fn arm_bx(&self, instr: &ArmDisasm) {
        todo!();
    }
}
