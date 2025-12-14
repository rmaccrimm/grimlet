use anyhow::Result;
use capstone::arch::arm::ArmCC;

use crate::arm::state::ArmState;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::jit::FunctionBuilder;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    // Either executes the branch built by `inner`, or increments the PC and exits the function
    fn exec_branch_conditional<F>(&mut self, instr: ArmInstruction, inner: F)
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<()>,
    {
        let mode = instr.mode;
        if instr.cond == ArmCC::ARM_CC_AL {
            // Expect that inner contains a return statement
            inner(self, instr).expect("LLVM codegen failed");
            return;
        }

        let build = |f: &mut Self| -> Result<()> {
            let ctx = f.llvm_ctx;
            let bd = f.builder;
            let if_block = ctx.append_basic_block(f.func, "if");
            let end_block = ctx.append_basic_block(f.func, "end");
            let cond = f.eval_cond(instr.cond)?;
            bd.build_conditional_branch(cond, if_block, end_block)?;
            bd.position_at_end(if_block);
            inner(f, instr)?;
            // No need to branch to end block, since we have returned already
            bd.position_at_end(end_block);
            f.increment_pc(mode);
            f.write_state_out()?;
            bd.build_return(None)?;
            Ok(())
        };
        build(self).expect("LLVM codegen failed");
    }

    pub(super) fn arm_b(&mut self, instr: ArmInstruction) {
        self.exec_branch_conditional(instr, Self::b);
    }

    fn b(&mut self, instr: ArmInstruction) -> Result<()> {
        let target = instr.get_imm_op(0) as usize;
        let bd = &self.builder;

        // TODO - backwards jumps
        match self.get_compiled_func_pointer(target)? {
            Some(_) => {
                // Not sure how/if we ever reach this yet
                todo!()
            }
            None => {
                // Context switch and jump out to the interpreter
                self.write_state_out()?;
                let func_ptr = self.get_external_func_pointer(
                    ArmState::jump_to as *const (&mut ArmState, u32) as usize,
                )?;
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
}
