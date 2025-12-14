use std::iter::zip;

use anyhow::Result;
use capstone::arch::arm::ArmCC;
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::jit::FunctionBuilder;

// A bitmask for a flag, and its name
#[derive(Copy, Clone, Debug)]
pub(super) struct Flag(pub u32, pub &'static str);

pub(super) const V: Flag = Flag(1 << 28, "v");
pub(super) const C: Flag = Flag(1 << 29, "c");
pub(super) const Z: Flag = Flag(1 << 30, "z");
pub(super) const N: Flag = Flag(1 << 31, "n");

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn eval_cond(&mut self, cond: ArmCC) -> Result<IntValue<'a>> {
        let bd = self.builder;
        let cond = match cond {
            ArmCC::ARM_CC_EQ => self.get_flag(Z)?,
            ArmCC::ARM_CC_NE => self.get_neg_flag(Z)?,
            ArmCC::ARM_CC_HS => self.get_flag(C)?,
            ArmCC::ARM_CC_LO => self.get_neg_flag(C)?,
            ArmCC::ARM_CC_MI => self.get_flag(N)?,
            ArmCC::ARM_CC_PL => self.get_neg_flag(N)?,
            ArmCC::ARM_CC_VS => self.get_flag(V)?,
            ArmCC::ARM_CC_VC => self.get_neg_flag(V)?,
            ArmCC::ARM_CC_HI => bd.build_and(self.get_flag(C)?, self.get_neg_flag(Z)?, "hi")?,
            ArmCC::ARM_CC_LS => bd.build_or(self.get_neg_flag(C)?, self.get_flag(Z)?, "ls")?,
            ArmCC::ARM_CC_GE => {
                bd.build_int_compare(IntPredicate::EQ, self.get_flag(N)?, self.get_flag(V)?, "ge")?
            }
            ArmCC::ARM_CC_LT => {
                bd.build_int_compare(IntPredicate::NE, self.get_flag(N)?, self.get_flag(V)?, "lt")?
            }
            ArmCC::ARM_CC_GT => bd.build_and(
                self.get_neg_flag(Z)?,
                bd.build_int_compare(IntPredicate::EQ, self.get_flag(N)?, self.get_flag(V)?, "ge")?,
                "gt",
            )?,
            ArmCC::ARM_CC_LE => bd.build_or(
                self.get_flag(Z)?,
                bd.build_int_compare(IntPredicate::NE, self.get_flag(N)?, self.get_flag(V)?, "lt")?,
                "le",
            )?,
            _ => panic!("invalid cond"),
        };
        Ok(cond)
    }

    /// Returns i1 IntValue
    pub(super) fn get_flag(&self, flag: Flag) -> Result<IntValue<'a>> {
        let masked = self.builder.build_and(
            self.reg_map.cpsr(),
            self.i32_t.const_int(flag.0 as u64, false),
            "flag",
        )?;
        Ok(self.builder.build_int_compare(
            IntPredicate::NE,
            self.i32_t.const_zero(),
            masked,
            flag.1,
        )?)
    }

    /// Returns i1 IntValue
    pub(super) fn get_neg_flag(&self, flag: Flag) -> Result<IntValue<'a>> {
        let f = self.get_flag(flag)?;
        let nf = self.builder.build_not(f, &format!("not_{}", flag.1))?;
        Ok(nf)
    }

    fn set_flag(
        &self,
        flag: Flag,
        initial: IntValue<'a>,
        cond: IntValue<'a>,
    ) -> Result<IntValue<'a>> {
        // From Stanford's bithacks page - conditionally set or clear bits without branching
        let bd = self.builder;
        let f = bd.build_int_cast_sign_flag(cond, self.i32_t, false, "f")?;
        let nf = bd.build_int_neg(f, "neg_f")?;
        let m = self.i32_t.const_int(flag.0 as u64, false);
        let lhs = bd.build_xor(nf, initial, "and_lhs")?;
        let and = bd.build_and(lhs, m, "and")?;
        let out = bd.build_xor(initial, and, &format!("set_{}", flag.1))?;
        Ok(out)
    }

    /// Return a copy of the current CPSR register with flags updated. Each flags value is
    /// expected to be an i1 (i.e. bool_t) IntValue.
    pub(super) fn set_flags(
        &self,
        n: Option<IntValue<'a>>,
        z: Option<IntValue<'a>>,
        c: Option<IntValue<'a>>,
        v: Option<IntValue<'a>>,
    ) -> Result<IntValue<'a>> {
        let mut cpsr = self.reg_map.cpsr();
        for (flag, cond) in zip([N, Z, C, V], [n, z, c, v]) {
            if let Some(v) = cond {
                cpsr = self.set_flag(flag, cpsr, v)?;
            }
        }
        Ok(cpsr)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use anyhow::Result;
    use inkwell::context::Context;

    use super::*;
    use crate::arm::state::{ArmState, REG_ITEMS, Reg};
    use crate::jit::Compiler;

    macro_rules! compile_and_run {
        ($compiler:ident, $func:ident, $state:ident) => {
            unsafe {
                $func.compile().unwrap().call(&mut $state);
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
        let mut comp = Compiler::new(&context);
        let mut func = comp.new_function(0, None);

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        func.load_initial_reg_values(&all_regs).unwrap();

        let bool_t = context.bool_type();
        let f = bool_t.const_zero();
        let t = bool_t.const_int(1, false);
        func.reg_map
            .update(Reg::R0, func.set_flag(V, func.reg_map.r0(), t)?);
        func.reg_map
            .update(Reg::R1, func.set_flag(V, func.reg_map.r1(), f)?);
        func.reg_map
            .update(Reg::R2, func.set_flag(V, func.reg_map.r2(), t)?);
        func.reg_map
            .update(Reg::R3, func.set_flag(V, func.reg_map.r3(), f)?);
        func.reg_map
            .update(Reg::R4, func.set_flag(N, func.reg_map.r4(), t)?);
        func.reg_map
            .update(Reg::R5, func.set_flag(N, func.reg_map.r5(), f)?);
        func.reg_map
            .update(Reg::R6, func.set_flag(N, func.reg_map.r6(), t)?);
        func.reg_map
            .update(Reg::R7, func.set_flag(N, func.reg_map.r7(), f)?);
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
