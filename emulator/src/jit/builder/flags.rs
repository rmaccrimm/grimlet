use anyhow::{Result, anyhow};
use capstone::arch::arm::{ArmCC, ArmInsn};
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::jit::FunctionBuilder;

#[derive(Copy, Clone, Debug)]
pub(super) enum Flag {
    V = 28,
    C = 29,
    Z = 30,
    N = 31,
}

impl Flag {
    pub fn to_str(self) -> &'static str {
        match self {
            Flag::V => "v",
            Flag::C => "c",
            Flag::Z => "z",
            Flag::N => "n",
        }
    }
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn get_cond_value(&mut self, cond: ArmCC) -> IntValue<'a> {
        let build = |f: &mut Self| -> Result<IntValue<'a>> {
            let bd = f.builder;
            let cond = match cond {
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
            Ok(cond)
        };
        build(self).expect("LLVM codegen failed")
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

    pub(super) fn set_flag(&self, flag: Flag, initial: IntValue, cond: IntValue) -> IntValue<'a> {
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use anyhow::Result;
    use inkwell::context::Context;

    use super::*;
    use crate::arm::cpu::ArmState;
    use crate::arm::disasm::cons::*;
    use crate::jit::Compiler;

    macro_rules! compile_and_run {
        ($compiler:ident, $func:ident, $state:ident) => {
            unsafe {
                let fptr = $func.compile().unwrap().as_raw();
                $compiler.compile_entry_point().call(&mut $state, fptr);
            }
        };
    }

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
}
