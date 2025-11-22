use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::{Compiler, LlvmFunction},
};
use anyhow::{Result, anyhow};
use capstone::arch::arm::{ArmInsn, ArmOperandType};
use inkwell::{IntPredicate, values::IntValue};

enum Flag {
    V = 28,
    C = 29,
    Z = 30,
    N = 31,
}

impl<'ctx> Compiler<'ctx> {
    fn set_flag(
        &self,
        func: &LlvmFunction,
        flag: Flag,
        initial: IntValue,
        cond: IntValue,
    ) -> Result<IntValue<'ctx>> {
        let ctx = &self.llvm_ctx;
        let bd = &self.builder;
        let if_block = ctx.append_basic_block(func.func, "if");
        let else_block = ctx.append_basic_block(func.func, "else");
        let end_block = ctx.append_basic_block(func.func, "else");

        let int1 = self.i32_t.const_int(1, false);
        let flag = self.i32_t.const_int(flag as u64, false);
        let result = bd.build_alloca(self.i32_t, "result")?;
        let v0 = bd.build_left_shift(int1, flag, "v0")?;
        bd.build_conditional_branch(cond, if_block, else_block)?;

        bd.position_at_end(if_block);
        let v1 = bd.build_and(v0, initial, "v1")?;
        bd.build_store(result, v1)?;
        bd.build_unconditional_branch(end_block)?;

        bd.position_at_end(else_block);
        let v3 = bd.build_not(v0, "v3")?;
        let v4 = bd.build_and(initial, v3, "v4")?;
        bd.build_store(result, v4)?;
        bd.build_unconditional_branch(end_block)?;

        bd.position_at_end(end_block);
        Ok(bd.build_load(self.i32_t, result, "v5")?.into_int_value())
    }

    /// Just z and n to start
    fn set_flags(&self, func: &mut LlvmFunction<'ctx>, last_value: IntValue) -> Result<()> {
        let bd = &self.builder;
        let z = bd.build_int_compare(IntPredicate::EQ, last_value, self.i32_t.const_zero(), "n")?;
        let n =
            bd.build_int_compare(IntPredicate::SLT, last_value, self.i32_t.const_zero(), "z")?;
        let cpsr0 = self.set_flag(func, Flag::Z, func.cpsr(), z)?;
        let cpsr1 = self.set_flag(func, Flag::N, cpsr0, n)?;
        func.update(Reg::CPSR, cpsr1);
        Ok(())
    }

    pub fn arm_cmp(&self, func: &mut LlvmFunction<'ctx>, instr: &ArmDisasm) -> Result<()> {
        let bd = &self.builder;
        let r = func.reg_map[instr.get_reg_op(0)?];
        let i = self.i32_t.const_int(instr.get_imm_op(1)? as u64, false);

        let v0 = bd.build_int_sub(r, i, "v0")?;
        self.set_flags(func, v0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::arm::cpu::ArmState;
    use inkwell::context::Context;

    use super::*;

    #[test]
    fn test_set_flags() {
        let mut state = ArmState::new();
        for i in [2, 3, 6, 7] {
            state.regs[i] = 0xff000000;
        }
        let context = Context::create();
        let mut comp = Compiler::new(&context).unwrap();
        let mut func = comp.new_function(0).unwrap();
        let r0 = func.reg_map[0];
        let t = comp.i32_t.const_zero();
        let f = comp.i32_t.const_int(1, false);
        func.update(
            Reg::R0,
            comp.set_flag(&func, Flag::V, func.r0(), t).unwrap(),
        );
        func.update(
            Reg::R1,
            comp.set_flag(&func, Flag::V, func.r1(), f).unwrap(),
        );
        func.update(
            Reg::R2,
            comp.set_flag(&func, Flag::V, func.r2(), t).unwrap(),
        );
        func.update(
            Reg::R3,
            comp.set_flag(&func, Flag::V, func.r3(), f).unwrap(),
        );
        func.update(
            Reg::R4,
            comp.set_flag(&func, Flag::N, func.r4(), t).unwrap(),
        );
        func.update(
            Reg::R5,
            comp.set_flag(&func, Flag::N, func.r5(), f).unwrap(),
        );
        func.update(
            Reg::R6,
            comp.set_flag(&func, Flag::N, func.r6(), t).unwrap(),
        );
        func.update(
            Reg::R7,
            comp.set_flag(&func, Flag::N, func.r7(), f).unwrap(),
        );
        comp.context_switch_out(&func).unwrap();
    }
}
