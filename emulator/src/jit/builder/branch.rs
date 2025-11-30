use anyhow::{Result, anyhow};
use inkwell::values::PointerValue;

use crate::{
    arm::{cpu::ArmState, disasm::ArmDisasm},
    jit::FunctionBuilder,
};

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    fn build_tail_call(&self, func_ptr: PointerValue) -> Result<()> {
        self.update_reg_array()?;
        let bd = &self.builder;
        let call = bd.build_indirect_call(
            self.fn_t,
            func_ptr,
            &[self.arm_state_ptr.into(), self.reg_array_ptr.into()],
            "call",
        )?;
        call.set_tail_call(true);
        bd.build_return(None)?;
        Ok(())
    }

    pub(super) fn arm_b(&self, instr: &ArmDisasm) -> Result<()> {
        let bd = &self.builder;
        let operand = instr.operands.first().ok_or(anyhow!("Bad operand"))?;
        let target = match operand.op_type {
            capstone::arch::arm::ArmOperandType::Imm(x) => x as usize,
            _ => {
                panic!("Bad operand")
            }
        };
        // TODO - backwards jumps?
        match self.get_compiled_func_pointer(target)? {
            Some(func_ptr) => {
                // Jump directly to compiled function
                self.build_tail_call(func_ptr)?;
            }
            None => {
                // Context switch and jump out to the interpreter
                self.write_state_out()?;
                let func_ptr = self.get_external_func_pointer(ArmState::jump_to as usize)?;
                let call = bd.build_indirect_call(
                    self.void_t
                        .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false),
                    func_ptr,
                    &[
                        self.arm_state_ptr.into(),
                        self.i32_t.const_int(target as u64, false).into(),
                    ],
                    "call",
                )?;
                call.set_tail_call(true);
                bd.build_return(None)?;
            }
        };
        Ok(())
    }

    fn arm_bl(&self, _instr: &ArmDisasm) {
        todo!();
    }

    fn arm_bx(&self, _instr: &ArmDisasm) {
        todo!();
    }
}
