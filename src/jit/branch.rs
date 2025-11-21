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
                self.context_switch_out(func.state_ptr, func.regs_ptr);
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
