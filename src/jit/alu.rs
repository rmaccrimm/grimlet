use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::LlvmFunction,
};
use anyhow::Result;
use inkwell::{IntPredicate, values::IntValue};

enum Flag {
    V = 28,
    C = 29,
    Z = 30,
    N = 31,
}

impl<'ctx, 'a> LlvmFunction<'ctx, 'a> {
    fn set_flag(&self, flag: Flag, initial: IntValue, cond: IntValue) -> Result<IntValue<'a>> {
        let ctx = &self.llvm_ctx;
        let bd = &self.builder;
        let if_block = ctx.append_basic_block(self.func, "if");
        let else_block = ctx.append_basic_block(self.func, "else");
        let end_block = ctx.append_basic_block(self.func, "else");

        let int1 = self.i32_t.const_int(1, false);
        let flag = self.i32_t.const_int(flag as u64, false);
        let result = bd.build_alloca(self.i32_t, "result")?;
        let v0 = bd.build_left_shift(int1, flag, "v0")?;
        bd.build_conditional_branch(cond, if_block, else_block)?;

        bd.position_at_end(if_block);
        let v1 = bd.build_or(v0, initial, "v1")?;
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
    fn set_flags(&mut self, last_value: IntValue) -> Result<()> {
        let bd = &self.builder;
        let z = bd.build_int_compare(IntPredicate::EQ, last_value, self.i32_t.const_zero(), "n")?;
        let n =
            bd.build_int_compare(IntPredicate::SLT, last_value, self.i32_t.const_zero(), "z")?;
        let cpsr0 = self.set_flag(Flag::Z, self.reg_map.cpsr(), z)?;
        let cpsr1 = self.set_flag(Flag::N, cpsr0, n)?;
        self.reg_map.update(Reg::CPSR, cpsr1);
        Ok(())
    }

    pub fn arm_cmp(&mut self, instr: &ArmDisasm) -> Result<()> {
        let bd = &self.builder;
        let r = self.reg_map.get(instr.get_reg_op(0)?);
        let i = self.i32_t.const_int(instr.get_imm_op(1)? as u64, false);

        let v0 = bd.build_int_sub(r, i, "v0")?;
        self.set_flags(v0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{arm::cpu::ArmState, jit::Compiler};
    use anyhow::Result;
    use inkwell::context::Context;

    use super::*;

    #[test]
    fn test_set_flags() -> Result<()> {
        let mut state = ArmState::default();
        for i in 0..8 {
            state.regs[i] = i as u32;
        }
        for i in [2, 3, 6, 7] {
            state.regs[i] |= 0xff000000;
        }
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context).unwrap();
        let mut func = comp.new_function(0, &cache).unwrap();

        let bool_t = context.bool_type();
        let f = bool_t.const_zero();
        let t = bool_t.const_int(1, false);
        func.reg_map
            .update(Reg::R0, func.set_flag(Flag::V, func.reg_map.r0(), t)?);
        func.reg_map
            .update(Reg::R1, func.set_flag(Flag::V, func.reg_map.r1(), f)?);
        func.reg_map
            .update(Reg::R2, func.set_flag(Flag::V, func.reg_map.r2(), t)?);
        func.reg_map
            .update(Reg::R3, func.set_flag(Flag::V, func.reg_map.r3(), f)?);
        func.reg_map
            .update(Reg::R4, func.set_flag(Flag::N, func.reg_map.r4(), t)?);
        func.reg_map
            .update(Reg::R5, func.set_flag(Flag::N, func.reg_map.r5(), f)?);
        func.reg_map
            .update(Reg::R6, func.set_flag(Flag::N, func.reg_map.r6(), t)?);
        func.reg_map
            .update(Reg::R7, func.set_flag(Flag::N, func.reg_map.r7(), f)?);
        func.write_state_out()?;

        compile_and_run!(comp, func, state);
        comp.dump();
        assert_eq!(
            &state.regs[0..8],
            [
                0x10_00_00_00,
                0x00_00_00_01,
                0xff_00_00_02,
                0xef_00_00_03,
                0x80_00_00_04,
                0x00_00_00_05,
                0xff_00_00_06,
                0x7f_00_00_07
            ]
        );
        Ok(())
    }
}
