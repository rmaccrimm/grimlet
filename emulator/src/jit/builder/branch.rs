use anyhow::Result;
use capstone::arch::arm::ArmCC;
use inkwell::basic_block::BasicBlock;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::ArmDisasm;
use crate::jit::FunctionBuilder;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    fn branch_internal(&mut self, instr: &ArmDisasm) -> Result<()> {
        // Not sure this is possible, given how CodeBlocks are terminated
        if instr.cond == ArmCC::ARM_CC_AL {
            todo!();
        }
        let ctx = self.llvm_ctx;
        let bd = self.builder;
        let target = instr.get_imm_op(0) as usize;
        let loop_block = *self.blocks.get(&target).expect("missing block");
        let end_block = ctx.append_basic_block(self.func, &format!("loop_{}_end", target));
        let cond = self.get_cond_value(instr.cond);
        bd.build_conditional_branch(cond, loop_block, end_block)?;
        bd.position_at_end(end_block);
        self.increment_pc(instr.mode);
        Ok(())
    }

    // Either executes the branch built by `inner`, or increments the PC and exits the function
    fn exec_branch_conditional<F>(&mut self, instr: &ArmDisasm, inner: F)
    where
        F: Fn(&mut Self) -> Result<()>,
    {
        if instr.cond == ArmCC::ARM_CC_AL {
            // Expect that inner contains a return statement
            inner(self).expect("LLVM codegen failed");
            return;
        }

        let build = |f: &mut Self| -> Result<()> {
            let ctx = f.llvm_ctx;
            let bd = f.builder;
            let if_block = ctx.append_basic_block(f.func, "if");
            let end_block = ctx.append_basic_block(f.func, "end");
            let cond = f.get_cond_value(instr.cond);
            bd.build_conditional_branch(cond, if_block, end_block)?;
            bd.position_at_end(if_block);
            inner(f)?;
            // No need to branch to end block, since we have returned already
            bd.position_at_end(end_block);
            f.increment_pc(instr.mode);
            f.write_state_out()?;
            bd.build_return(None)?;
            Ok(())
        };
        build(self).expect("LLVM codegen failed");
    }

    pub(super) fn arm_b(&mut self, instr: &ArmDisasm) {
        let target = instr.get_imm_op(0) as usize;
        if self.blocks.contains_key(&target) {
            self.branch_internal(instr).expect("LLVM codegen failed");
            return;
        }

        let build = |f: &mut Self| -> Result<()> {
            let bd = &f.builder;

            // TODO - backwards jumps
            match f.get_compiled_func_pointer(target)? {
                Some(func_ptr) => {
                    // Jump directly to compiled function
                    f.update_reg_array()?;
                    let call = bd.build_indirect_call(
                        f.fn_t,
                        func_ptr,
                        &[f.arm_state_ptr.into(), f.reg_array_ptr.into()],
                        "call",
                    )?;
                    call.set_tail_call(true);
                    bd.build_return(None)?;
                }
                None => {
                    // Context switch and jump out to the interpreter
                    f.write_state_out()?;
                    let func_ptr = f.get_external_func_pointer(
                        ArmState::jump_to as *const (&mut ArmState, u32) as usize,
                    )?;
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
        self.exec_branch_conditional(instr, build);
    }

    fn arm_bl(&self, _instr: &ArmDisasm) {
        todo!();
    }

    fn arm_bx(&self, _instr: &ArmDisasm) {
        todo!();
    }
}
