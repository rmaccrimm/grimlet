use anyhow::{Context as _, Result};
use inkwell::values::IntValue;

use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::{ArmState, Reg};
use crate::jit::FunctionBuilder;

pub(super) struct BranchAction<'a> {
    pub(super) target: IntValue<'a>,
    pub(super) save_return: bool,
    pub(super) change_mode: Option<IntValue<'a>>,
}

impl<'a> FunctionBuilder<'_, 'a> {
    pub(super) fn arm_b(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::b);
    }

    pub(super) fn arm_bl(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::bl);
    }

    pub(super) fn arm_bx(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_branch_conditional, instr, Self::bx);
    }

    // Either executes the branch built by `inner`, or increments the PC and exits the function
    fn exec_branch_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&Self, &ArmInstruction) -> Result<BranchAction<'a>>,
    {
        let bd = &self.builder;
        let mode = instr.mode;
        let ctx = self.ctx;
        let if_block = ctx.append_basic_block(self.func, "if");
        let end_block = ctx.append_basic_block(self.func, "end");
        let cond = self.eval_cond(instr.cond)?;

        // This block just so we can have the temporary builder borrow

        bd.build_conditional_branch(cond, if_block, end_block)?;
        bd.position_at_end(if_block);

        let BranchAction {
            target,
            save_return,
            change_mode,
        } = inner(self, instr)?;

        let mode_param = match change_mode {
            Some(mode) => mode,
            None => {
                // Keep mode the same as current
                imm8!(self, instr.mode as i8)
            }
        };

        if save_return {
            let return_addr = imm!(self, instr.addr + instr.size);
            // Update lr temporarily, and restore it after the context switch so that
            // the unaltered value is used if we skip the end block
            let mut tmp_reg_map = self.reg_map.clone();
            tmp_reg_map.update(Reg::LR, return_addr);
            self.write_state_out(&tmp_reg_map)?;
        } else {
            self.write_state_out(&self.reg_map)?;
        }
        self.branch_and_return(target, mode_param)?;

        // No need to branch to end block, since we have returned already
        bd.position_at_end(end_block);
        self.increment_pc(instr.size);
        self.write_state_out(&self.reg_map)?;

        self.builder.build_return(None)?;
        Ok(())
    }

    // self.write_state_out should be called before calling this
    pub(super) fn branch_and_return(&self, target: IntValue<'a>, mode: IntValue<'a>) -> Result<()> {
        let bd = &self.builder;
        let func_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i8_t.into()],
            false,
        );
        let func_ptr = self
            .get_external_func_pointer(ArmState::jump_to as fn(&mut ArmState, u32, i8) as usize)?;

        let call = call_indirect!(bd, func_t, func_ptr, self.arm_state_ptr, target, mode);
        call.set_tail_call(true);
        bd.build_return(None)?;
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn b(&self, instr: &ArmInstruction) -> Result<BranchAction<'a>> {
        Ok(BranchAction {
            target: imm!(self, instr.get_imm_op(0)),
            save_return: false,
            change_mode: None,
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn bl(&self, instr: &ArmInstruction) -> Result<BranchAction<'a>> {
        Ok(BranchAction {
            target: imm!(self, instr.get_imm_op(0)),
            save_return: true,
            change_mode: None,
        })
    }

    fn bx(&self, instr: &ArmInstruction) -> Result<BranchAction<'a>> {
        let bd = &self.builder;
        let rm_val = self.reg_map.get(instr.get_reg_op(0));
        let lsb = bd.build_and(rm_val, imm!(self, 1), "lsb")?;
        let mode = bd.build_int_cast_sign_flag(lsb, self.i8_t, false, "mode")?;

        let mask = bd.build_not(imm!(self, 1), "mask")?;
        let target = bd.build_and(rm_val, mask, "target")?;

        Ok(BranchAction {
            target,
            save_return: false,
            change_mode: Some(mode),
        })
    }
}
