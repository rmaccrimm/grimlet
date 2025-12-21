use anyhow::{Context as _, Result as AnyResult};
use inkwell::values::IntValue;

use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::{ArmState, Reg};
use crate::jit::FunctionBuilder;

pub(super) struct BranchAction<'a> {
    target: IntValue<'a>,
    save_return: bool,
    change_mode: bool,
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    // Either executes the branch built by `inner`, or increments the PC and exits the function
    fn exec_branch_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> AnyResult<()>
    where
        F: Fn(&Self, &ArmInstruction) -> AnyResult<BranchAction<'a>>,
    {
        let mode = instr.mode;
        let ctx = self.llvm_ctx;
        let bd = self.builder;
        let if_block = ctx.append_basic_block(self.func, "if");
        let end_block = ctx.append_basic_block(self.func, "end");
        let cond = self.eval_cond(instr.cond)?;
        bd.build_conditional_branch(cond, if_block, end_block)?;
        bd.position_at_end(if_block);

        let branch = inner(self, instr)?;
        if branch.save_return {
            let return_addr = bd.build_int_sub(self.reg_map.get(Reg::PC), imm!(self, 1), "ret")?;
            // Update lr temporarily, and restore it after the context switch transfer so that
            // the unaltered value is used if we skip the end block
            let prev_lr = self.reg_map.get(Reg::LR);
            self.reg_map.update(Reg::LR, return_addr);
            self.write_state_out()?;
            self.reg_map.update(Reg::LR, prev_lr);
        } else {
            self.write_state_out()?;
        }
        let func_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i8_t.into()],
            false,
        );

        let func_ptr = self.get_external_func_pointer(
            ArmState::jump_to as fn(&mut ArmState, u32, bool) as usize,
        )?;
        let call = call_indirect!(
            bd,
            func_t,
            func_ptr,
            self.arm_state_ptr,
            branch.target,
            imm8!(self, branch.change_mode)
        );
        call.set_tail_call(true);
        bd.build_return(None)?;

        // No need to branch to end block, since we have returned already
        bd.position_at_end(end_block);
        self.increment_pc(mode);
        self.write_state_out()?;
        bd.build_return(None)?;
        Ok(())
    }

    pub(super) fn arm_b(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::b);
    }

    pub(super) fn arm_bl(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::bl);
    }

    pub(super) fn arm_bx(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::bx);
    }

    fn b(&self, instr: &ArmInstruction) -> AnyResult<BranchAction<'a>> {
        Ok(BranchAction {
            target: imm!(self, instr.get_imm_op(0)),
            save_return: false,
            change_mode: false,
        })
    }

    fn bl(&self, instr: &ArmInstruction) -> AnyResult<BranchAction<'a>> {
        Ok(BranchAction {
            target: imm!(self, instr.get_imm_op(0)),
            save_return: true,
            change_mode: false,
        })
    }

    fn bx(&self, instr: &ArmInstruction) -> AnyResult<BranchAction<'a>> {
        // TODO reg
        Ok(BranchAction {
            target: self.reg_map.get(instr.get_reg_op(0)),
            save_return: false,
            change_mode: true,
        })
    }
}
