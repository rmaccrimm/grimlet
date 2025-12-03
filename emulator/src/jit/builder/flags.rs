use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::FunctionBuilder,
};
use anyhow::{Result, anyhow};
use capstone::arch::arm::{ArmCC, ArmInsn};
use inkwell::{IntPredicate, values::IntValue};

#[derive(Copy, Clone, Debug)]
enum Flag {
    V = 28,
    C = 29,
    Z = 30,
    N = 31,
}

impl Flag {
    fn to_str(self) -> &'static str {
        match self {
            Flag::V => "v",
            Flag::C => "c",
            Flag::Z => "z",
            Flag::N => "n",
        }
    }
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn exec_conditional<F>(&mut self, instr: &ArmDisasm, func: F, branch: bool)
    where
        F: Fn(&mut Self) -> Result<()>,
    {
        if instr.cond == ArmCC::ARM_CC_AL {
            func(self).expect("LLVM codegen failed");
            return;
        }
        self.compute_flags();
        let build = |f: &mut Self| -> Result<()> {
            let ctx = f.llvm_ctx;
            let bd = f.builder;
            let if_block = ctx.append_basic_block(f.func, "if");
            let end_block = ctx.append_basic_block(f.func, "end");
            let cond = match instr.cond {
                ArmCC::ARM_CC_EQ => f.get_flag(Flag::Z),
                ArmCC::ARM_CC_NE => f.get_neg_flag(Flag::Z),
                ArmCC::ARM_CC_HS => f.get_flag(Flag::C),
                ArmCC::ARM_CC_LO => f.get_neg_flag(Flag::C),
                ArmCC::ARM_CC_MI => f.get_flag(Flag::N),
                ArmCC::ARM_CC_PL => f.get_neg_flag(Flag::N),
                ArmCC::ARM_CC_VS => f.get_flag(Flag::V),
                ArmCC::ARM_CC_VC => f.get_neg_flag(Flag::V),
                ArmCC::ARM_CC_HI => {
                    bd.build_and(f.get_flag(Flag::C), f.get_neg_flag(Flag::Z), "hi")?
                }
                ArmCC::ARM_CC_LS => {
                    bd.build_or(f.get_neg_flag(Flag::C), f.get_flag(Flag::Z), "ls")?
                }
                ArmCC::ARM_CC_GE => bd.build_int_compare(
                    IntPredicate::EQ,
                    f.get_flag(Flag::N),
                    f.get_flag(Flag::V),
                    "ge",
                )?,
                ArmCC::ARM_CC_LT => bd.build_int_compare(
                    IntPredicate::NE,
                    f.get_flag(Flag::N),
                    f.get_flag(Flag::V),
                    "lt",
                )?,
                ArmCC::ARM_CC_GT => bd.build_and(
                    f.get_neg_flag(Flag::Z),
                    bd.build_int_compare(
                        IntPredicate::EQ,
                        f.get_flag(Flag::N),
                        f.get_flag(Flag::V),
                        "ge",
                    )?,
                    "gt",
                )?,
                ArmCC::ARM_CC_LE => bd.build_or(
                    f.get_flag(Flag::Z),
                    bd.build_int_compare(
                        IntPredicate::NE,
                        f.get_flag(Flag::N),
                        f.get_flag(Flag::V),
                        "lt",
                    )?,
                    "le",
                )?,
                _ => panic!("invalid cond"),
            };
            bd.build_conditional_branch(cond, if_block, end_block)?;
            bd.position_at_end(if_block);
            func(f)?;
            if branch {
                bd.build_unconditional_branch(end_block)?;
            }
            bd.position_at_end(end_block);
            Ok(())
        };
        build(self).expect("LLVM codegen failed");
    }

    fn get_flag(&self, flag: Flag) -> IntValue<'a> {
        let build = || -> Result<IntValue> {
            let bd = self.builder;
            let shift = bd.build_right_shift(
                self.reg_map.cpsr(),
                self.i32_t.const_int(flag as u64, false),
                false,
                "shift",
            )?;
            let flag_i32 = bd.build_and(shift, self.i32_t.const_int(1, false), "flag")?;
            Ok(bd.build_int_cast(flag_i32, self.bool_t, flag.to_str())?)
        };
        build().expect("LLVM codegen failed")
    }

    fn get_neg_flag(&self, flag: Flag) -> IntValue<'a> {
        self.builder
            .build_not(self.get_flag(flag), &format!("not_{}", flag.to_str()))
            .expect("LLVM codegen failed")
    }

    fn set_flag(&self, flag: Flag, initial: IntValue, cond: IntValue) -> IntValue<'a> {
        let ctx = &self.llvm_ctx;
        let bd = &self.builder;
        let if_block = ctx.append_basic_block(self.func, "if");
        let else_block = ctx.append_basic_block(self.func, "else");
        let end_block = ctx.append_basic_block(self.func, "end");

        let int1 = self.i32_t.const_int(1, false);
        let flag = self.i32_t.const_int(flag as u64, false);

        let build = || -> Result<IntValue> {
            let v0 = bd.build_left_shift(int1, flag, "v0")?;
            bd.build_conditional_branch(cond, if_block, else_block)?;

            bd.position_at_end(if_block);
            let v1 = bd.build_or(v0, initial, "v1")?;
            // bd.build_store(result, v1)?;
            bd.build_unconditional_branch(end_block)?;

            bd.position_at_end(else_block);
            let v3 = bd.build_not(v0, "v3")?;
            let v4 = bd.build_and(initial, v3, "v4")?;
            // bd.build_store(result, v4)?;
            bd.build_unconditional_branch(end_block)?;

            bd.position_at_end(end_block);
            let phi = bd.build_phi(self.i32_t, "phi")?;
            phi.add_incoming(&[(&v1, if_block), (&v4, else_block)]);
            Ok(phi.as_basic_value().into_int_value())
        };
        build().expect("LLVM codegen failed")
    }

    fn compute_flags(&mut self) {
        // C and V - use LLVM intrinsic and re-run last instruction with overflow check
        let cpsr_next = match self.last_instr.opcode {
            ArmInsn::ARM_INS_CMP => self.compute_sub_flags(),
            ArmInsn::ARM_INS_NOP => self.reg_map.cpsr(),
            _ => todo!(),
        };
        self.reg_map.update(Reg::CPSR, cpsr_next);
    }

    fn compute_sub_flags(&mut self) -> IntValue<'a> {
        let mut build = || -> Result<IntValue> {
            let mut cpsr = self.reg_map.cpsr();
            let bd = &self.builder;
            let imm = self
                .last_instr
                .inputs
                .pop()
                .ok_or(anyhow!("Missing operand"))?;
            let reg_val = self
                .last_instr
                .inputs
                .pop()
                .ok_or(anyhow!("Missing operand"))?;

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

            let n_flag =
                bd.build_int_compare(IntPredicate::SLT, sres_val, self.i32_t.const_zero(), "z")?;

            cpsr = self.set_flag(Flag::C, cpsr, c_flag);
            cpsr = self.set_flag(Flag::Z, cpsr, z_flag);
            cpsr = self.set_flag(Flag::N, cpsr, n_flag);
            cpsr = self.set_flag(Flag::V, cpsr, v_flag);
            Ok(cpsr)
        };
        build().expect("LLVM codegen failed")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::jit::compile_and_run;
    use crate::{
        arm::{cpu::ArmState, disasm::cons::*},
        jit::Compiler,
    };
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
        let mut comp = Compiler::new(&context);
        let mut func = comp.new_function(0, &cache);

        let f = func.bool_t.const_zero();
        let t = func.bool_t.const_int(1, false);
        func.reg_map
            .update(Reg::R0, func.set_flag(Flag::V, func.reg_map.r0(), t));
        func.reg_map
            .update(Reg::R1, func.set_flag(Flag::V, func.reg_map.r1(), f));
        func.reg_map
            .update(Reg::R2, func.set_flag(Flag::V, func.reg_map.r2(), t));
        func.reg_map
            .update(Reg::R3, func.set_flag(Flag::V, func.reg_map.r3(), f));
        func.reg_map
            .update(Reg::R4, func.set_flag(Flag::N, func.reg_map.r4(), t));
        func.reg_map
            .update(Reg::R5, func.set_flag(Flag::N, func.reg_map.r5(), f));
        func.reg_map
            .update(Reg::R6, func.set_flag(Flag::N, func.reg_map.r6(), t));
        func.reg_map
            .update(Reg::R7, func.set_flag(Flag::N, func.reg_map.r7(), f));
        func.write_state_out().unwrap();
        func.builder.build_return(None).unwrap();
        compile_and_run!(comp, func, state);
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
    fn test_compute_flags_cmp() {
        let cmp_instr = op_reg_imm(ArmInsn::ARM_INS_CMP, 0, 1, None);

        let mut state = ArmState::default();
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context);

        let mut f1 = comp.new_function(0, &cache);
        f1.build(&cmp_instr);
        f1.compute_flags();
        f1.write_state_out().unwrap();
        f1.builder.build_return(None).unwrap();
        let cmp = f1.compile();
        let entry_point = comp.compile_entry_point();

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
    }

    #[test]
    fn test_conditions() {
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context);

        let mut test_case = |cond: ArmCC, flags: u32, should_execute: bool| {
            let mut state = ArmState::default();
            state.regs[Reg::CPSR as usize] = flags << 28;
            let mut f = comp.new_function(0, &cache);
            f.build(&op_imm(ArmInsn::ARM_INS_B, 100, Some(cond)));
            f.build(&op_imm(ArmInsn::ARM_INS_B, 200, None));
            compile_and_run!(comp, f, state);
            assert_eq!(state.pc(), if should_execute { 100 } else { 200 });
        };
        //                            _z__
        test_case(ArmCC::ARM_CC_EQ, 0b0001, false);
        test_case(ArmCC::ARM_CC_EQ, 0b0111, true);
        test_case(ArmCC::ARM_CC_NE, 0b1000, true);
        test_case(ArmCC::ARM_CC_NE, 0b1110, false);
        //                            __c_
        test_case(ArmCC::ARM_CC_HS, 0b0100, false);
        test_case(ArmCC::ARM_CC_HS, 0b1010, true);
        test_case(ArmCC::ARM_CC_LO, 0b1001, true);
        test_case(ArmCC::ARM_CC_LO, 0b1110, false);
        //                            n___
        test_case(ArmCC::ARM_CC_MI, 0b0000, false);
        test_case(ArmCC::ARM_CC_MI, 0b1001, true);
        test_case(ArmCC::ARM_CC_PL, 0b0000, true);
        test_case(ArmCC::ARM_CC_PL, 0b1000, false);
        //                            ___v
        test_case(ArmCC::ARM_CC_VS, 0b0010, false);
        test_case(ArmCC::ARM_CC_VS, 0b1001, true);
        test_case(ArmCC::ARM_CC_VC, 0b1100, true);
        test_case(ArmCC::ARM_CC_VC, 0b1101, false);
        //                            _zc_
        test_case(ArmCC::ARM_CC_HI, 0b0001, false);
        test_case(ArmCC::ARM_CC_HI, 0b0011, true);
        test_case(ArmCC::ARM_CC_HI, 0b1100, false);
        test_case(ArmCC::ARM_CC_HI, 0b0110, false);
        test_case(ArmCC::ARM_CC_LS, 0b0000, true);
        test_case(ArmCC::ARM_CC_LS, 0b0010, false);
        test_case(ArmCC::ARM_CC_LS, 0b0100, true);
        test_case(ArmCC::ARM_CC_LS, 0b0110, true);
        //                            n__v
        test_case(ArmCC::ARM_CC_GE, 0b0000, true);
        test_case(ArmCC::ARM_CC_GE, 0b0001, false);
        test_case(ArmCC::ARM_CC_GE, 0b1000, false);
        test_case(ArmCC::ARM_CC_GE, 0b1001, true);
        test_case(ArmCC::ARM_CC_LT, 0b0000, false);
        test_case(ArmCC::ARM_CC_LT, 0b0001, true);
        test_case(ArmCC::ARM_CC_LT, 0b1000, true);
        test_case(ArmCC::ARM_CC_LT, 0b1001, false);
        //                            nz_v
        test_case(ArmCC::ARM_CC_GT, 0b0010, true);
        test_case(ArmCC::ARM_CC_GT, 0b0001, false);
        test_case(ArmCC::ARM_CC_GT, 0b1000, false);
        test_case(ArmCC::ARM_CC_GT, 0b1011, true);
        test_case(ArmCC::ARM_CC_GT, 0b0110, false);
        test_case(ArmCC::ARM_CC_GT, 0b0101, false);
        test_case(ArmCC::ARM_CC_GT, 0b1100, false);
        test_case(ArmCC::ARM_CC_GT, 0b1111, false);
        //                            nz_v
        test_case(ArmCC::ARM_CC_LE, 0b0000, false);
        test_case(ArmCC::ARM_CC_LE, 0b0001, true);
        test_case(ArmCC::ARM_CC_LE, 0b1000, true);
        test_case(ArmCC::ARM_CC_LE, 0b1011, false);
        test_case(ArmCC::ARM_CC_LE, 0b0110, true);
        test_case(ArmCC::ARM_CC_LE, 0b0101, true);
        test_case(ArmCC::ARM_CC_LE, 0b1100, true);
        test_case(ArmCC::ARM_CC_LE, 0b1101, true);
    }
}
