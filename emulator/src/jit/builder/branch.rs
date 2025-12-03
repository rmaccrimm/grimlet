use anyhow::Result;
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

    pub(super) fn arm_b(&mut self, instr: &ArmDisasm) {
        let build = |f: &mut Self| -> Result<()> {
            let bd = &f.builder;
            let target = instr.get_imm_op(0) as usize;
            // TODO - backwards jumps?
            match f.get_compiled_func_pointer(target)? {
                Some(func_ptr) => {
                    // Jump directly to compiled function
                    f.build_tail_call(func_ptr)?;
                }
                None => {
                    // Context switch and jump out to the interpreter
                    f.write_state_out()?;
                    let func_ptr = f.get_external_func_pointer(ArmState::jump_to as usize)?;
                    let call = bd.build_indirect_call(
                        f.void_t.fn_type(&[f.ptr_t.into(), f.i32_t.into()], false),
                        func_ptr,
                        &[
                            f.arm_state_ptr.into(),
                            f.i32_t.const_int(target as u64, false).into(),
                        ],
                        "call",
                    )?;
                    call.set_tail_call(true);
                    bd.build_return(None)?;
                }
            };
            Ok(())
        };
        self.exec_conditional(instr, build, false);
    }

    fn arm_bl(&self, _instr: &ArmDisasm) {
        todo!();
    }

    fn arm_bx(&self, _instr: &ArmDisasm) {
        todo!();
    }
}
