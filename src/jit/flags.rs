use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::LlvmFunction,
};
use anyhow::Result;
use capstone::arch::arm::ArmInsn;
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

    fn compute_flags(&mut self) -> Result<()> {
        let bd = &self.builder;
        let mut cpsr = self.reg_map.cpsr();

        // C and V - use LLVM intrinsic and re-run last instruction with overflow check
        match self.last_instr.opcode {
            ArmInsn::ARM_INS_CMP => {
                let reg_val = self.reg_map.get(self.last_instr.get_reg_op(0)?);
                let imm = self
                    .i32_t
                    .const_int(self.last_instr.get_imm_op(1)? as u64, false);

                let ures = bd
                    .build_call(
                        self.usub_with_overflow,
                        &[reg_val.into(), imm.into()],
                        "ures",
                    )?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_struct_value();

                let sres = bd
                    .build_call(
                        self.ssub_with_overflow,
                        &[reg_val.into(), imm.into()],
                        "sres",
                    )?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_struct_value();

                let sres_val = bd
                    .build_extract_value(sres, 0, "sres_val")?
                    .into_int_value();

                let v_flag = bd.build_not(
                    bd.build_extract_value(sres, 1, "not_v")?.into_int_value(),
                    "v",
                )?;
                let c_flag = bd.build_not(
                    bd.build_extract_value(ures, 1, "not_c")?.into_int_value(),
                    "c",
                )?;

                let z_flag =
                    bd.build_int_compare(IntPredicate::EQ, sres_val, self.i32_t.const_zero(), "n")?;

                let n_flag = bd.build_int_compare(
                    IntPredicate::SLT,
                    sres_val,
                    self.i32_t.const_zero(),
                    "z",
                )?;
                cpsr = self.set_flag(Flag::C, cpsr, c_flag)?;
                cpsr = self.set_flag(Flag::Z, cpsr, z_flag)?;
                cpsr = self.set_flag(Flag::N, cpsr, n_flag)?;
                cpsr = self.set_flag(Flag::V, cpsr, v_flag)?;
            }
            _ => todo!(),
        };

        self.reg_map.update(Reg::CPSR, cpsr);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{arm::cpu::ArmState, jit::Compiler};
    use anyhow::Result;
    use capstone::{
        RegId,
        arch::arm::{ArmOperand, ArmOperandType},
    };
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
        comp.dump()?;
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

    #[test]
    fn test_compute_flags_cmp() -> Result<()> {
        let mut state = ArmState::default();
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context).unwrap();
        let mut f1 = comp.new_function(0, &cache).unwrap();
        // CMP with 1
        f1.last_instr = ArmDisasm {
            opcode: ArmInsn::ARM_INS_CMP,
            operands: vec![
                ArmOperand {
                    op_type: ArmOperandType::Reg(RegId(0)),
                    ..ArmOperand::default()
                },
                ArmOperand {
                    op_type: ArmOperandType::Imm(1),
                    ..ArmOperand::default()
                },
            ],
            ..ArmDisasm::default()
        };
        f1.compute_flags()?;
        f1.write_state_out()?;
        let cmp = f1.compile()?;
        comp.dump()?;
        let entry_point = comp.compile_entry_point()?;

        // Positive result
        state.regs[0] = 2;
        state.regs[16] = 0;
        unsafe {
            entry_point.call(&mut state, cmp.as_raw());
        }
        assert_eq!(state.regs[16] >> 28, 0b0011); // nzcv

        // 0 result
        state.regs[0] = 1;
        state.regs[16] = 0;
        unsafe {
            entry_point.call(&mut state, cmp.as_raw());
        }
        assert_eq!(state.regs[16] >> 28, 0b0111);

        // negative result (unsigned underflow)
        state.regs[0] = 0;
        state.regs[16] = 0;
        unsafe {
            entry_point.call(&mut state, cmp.as_raw());
        }
        assert_eq!(state.regs[16] >> 28, 0b1001);

        // negative result (no underflow)
        state.regs[0] = -1i32 as u32;
        state.regs[16] = 0;
        unsafe {
            entry_point.call(&mut state, cmp.as_raw());
        }
        assert_eq!(state.regs[16] >> 28, 0b1011);

        // signed underflow only (positive result)
        state.regs[0] = i32::MIN as u32;
        state.regs[16] = 0;
        unsafe {
            entry_point.call(&mut state, cmp.as_raw());
        }
        println!("{}", state.regs[0]);
        assert_eq!(state.regs[16] >> 28, 0b0010);

        Ok(())
    }
}
