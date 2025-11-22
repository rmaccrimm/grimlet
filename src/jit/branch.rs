use anyhow::{Result, anyhow};
use inkwell::values::PointerValue;

use crate::{
    arm::{cpu::ArmState, disasm::ArmDisasm},
    jit::{Compiler, FuncCacheKey, LlvmFunction},
};

impl<'ctx> Compiler<'ctx> {
    pub fn arm_b(&self, func: &LlvmFunction, instr: &ArmDisasm) -> Result<()> {
        let bd = &self.builder;
        let operand = instr.operands.iter().next().ok_or(anyhow!("Bad operand"))?;
        let target = match operand.op_type {
            capstone::arch::arm::ArmOperandType::Imm(x) => x,
            _ => {
                panic!("Bad operand")
            }
        };
        // TODO - backwards jumps?
        match self.get_compiled_func_pointer(FuncCacheKey(target as u32))? {
            Some(func_ptr) => {
                // Jump directly to compiled function
                let call = bd.build_indirect_call(
                    self.fn_t,
                    func_ptr,
                    &[func.state_ptr.into(), func.regs_ptr.into()],
                    "call",
                )?;
                call.set_tail_call(true);
                bd.build_return(None)?;
            }
            None => {
                // Context switch and jump out to the interpreter
                self.context_switch_out(&func)?;
                let func_ptr = self.get_external_func_pointer(ArmState::jump_to as u64)?;
                let call = bd.build_indirect_call(
                    self.void_t
                        .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false),
                    func_ptr,
                    &[
                        func.state_ptr.into(),
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

    pub fn arm_bl(&self, instr: &ArmDisasm) {
        todo!();
    }

    pub fn arm_bx(&self, instr: &ArmDisasm) {
        todo!();
    }
}
