use anyhow::{Result, anyhow};
use capstone::arch::arm::ArmInsn;
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::disasm::instruction::{ArmInstruction, ArmShift, ShifterOperand};
use crate::arm::state::{ArmMode, Reg};
use crate::jit::flags::C;
use crate::jit::{FunctionBuilder, InstrEffect, InstrResult, RegUpdate};

fn get_reg_shift(op: ArmInsn) -> impl Fn(Reg) -> ArmShift {
    match op {
        ArmInsn::ARM_INS_ASR => ArmShift::AsrReg,
        ArmInsn::ARM_INS_LSR => ArmShift::LsrReg,
        ArmInsn::ARM_INS_LSL => ArmShift::LslReg,
        ArmInsn::ARM_INS_ROR => ArmShift::RorReg,
        _ => panic!("instr is not a shift: {op:?}"),
    }
}

fn get_imm_shift(op: ArmInsn) -> impl Fn(u32) -> ArmShift {
    match op {
        ArmInsn::ARM_INS_ASR => ArmShift::AsrImm,
        ArmInsn::ARM_INS_LSR => ArmShift::LsrImm,
        ArmInsn::ARM_INS_LSL => ArmShift::LslImm,
        ArmInsn::ARM_INS_ROR => ArmShift::RorImm,
        _ => panic!("instr is not a shift: {op:?}"),
    }
}

impl<'a> FunctionBuilder<'_, 'a> {
    // Returns (i32, Option<i1>) - the value of the operand and the shifter carry-out value (None if
    // unaffected)
    pub(super) fn shift_value(
        &self,
        base: IntValue<'a>,
        shift: Option<ArmShift>,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        match shift {
            None => Ok((base, None)),
            Some(ArmShift::LslImm(imm)) => self.lsl_imm(base, imm),
            Some(ArmShift::LslReg(reg_id)) => self.lsl_reg(base, reg_id),
            Some(ArmShift::LsrImm(imm)) => self.lsr_imm(base, imm),
            Some(ArmShift::LsrReg(reg_id)) => self.lsr_reg(base, reg_id),
            Some(ArmShift::AsrImm(imm)) => self.asr_imm(base, imm),
            Some(ArmShift::AsrReg(reg_id)) => self.asr_reg(base, reg_id),
            Some(ArmShift::RorImm(imm)) => self.ror_imm(base, imm),
            Some(ArmShift::RorReg(reg_id)) => self.ror_reg(base, reg_id),
            Some(ArmShift::Rrx) => self.rrx_imm(base),
        }
    }

    pub(super) fn shifter_operand(
        &self,
        operand: ShifterOperand,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;

        match operand {
            ShifterOperand::Imm { imm, rotate } => {
                if let Some(r) = rotate {
                    let imm_val = imm!(self, imm);
                    let rot_amt = imm!(self, r);
                    let rot_val =
                        call_intrinsic!(bd, self.fshr, imm_val, imm_val, rot_amt).into_int_value();
                    let shifted = bd.build_right_shift(rot_val, imm!(self, 31), false, "sh")?;
                    let rot_msb =
                        bd.build_int_compare(IntPredicate::EQ, shifted, imm!(self, 1), "rot_msb")?;
                    Ok((rot_val, Some(rot_msb)))
                } else {
                    Ok((imm!(self, imm), None))
                }
            }
            ShifterOperand::Reg { reg, shift } => {
                let base = self.reg_map.get(reg);
                let shifted = self.shift_value(base, shift)?;
                Ok(shifted)
            }
        }
    }

    pub(super) fn shift_op(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rd_val = self.reg_map.get(rd);

        let (sh_val, c) = {
            // Technically ARMv4 does not support supported shift instructions in ARM mode, but
            // Capstone decodes some mov instructions with a shifter operand to a shift instruction
            // e.g.
            // `mov r1, r1, rrx` is decoded as `rrx r1, r1`
            // `movs r10, r1, lsl r12` is decoded as `lsls sl, r1, ip`
            //
            // In these cases, we don't have the shift type encoded in an operand, so we go back
            // to the opcode again (get_imm_shift/get_reg_shift).
            match instr.mode {
                ArmMode::ARM => {
                    let num_operands = instr.operands.len();
                    debug_assert!((2..=3).contains(&num_operands));
                    // immediate shifts have only two operands, with second being a shifter operand
                    // (just a mov), registers have 3 (no shifter operand)
                    if num_operands == 2 {
                        return self.mov(instr);
                    }
                    let rm = instr.get_reg_op(1);
                    let rm_val = self.reg_map.get(rm);
                    let rs = instr.get_reg_op(2);
                    self.shift_value(rm_val, Some(get_reg_shift(instr.opcode)(rs)))?
                }
                ArmMode::THUMB => {
                    let num_operands = instr.operands.len();
                    debug_assert!((2..=3).contains(&num_operands));
                    // immediate shifts come through with 3 operands (rd, rd, #imm) while by reg
                    // comes through with 2 (rd, rm)
                    if num_operands == 2 {
                        let rm = instr.get_reg_op(1);
                        let rm_val = self.reg_map.get(rm);
                        let shift_reg = instr.get_reg_op(2);
                        self.shift_value(rm_val, Some(get_reg_shift(instr.opcode)(shift_reg)))?
                    } else {
                        let shift_amt = instr.get_imm_op(2);
                        self.shift_value(
                            rd_val,
                            Some(get_imm_shift(instr.opcode)(shift_amt.cast_unsigned())),
                        )?
                    }
                }
            }
        };
        let n = bd.build_int_compare(IntPredicate::SLT, sh_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, sh_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        let updates = vec![RegUpdate(rd, sh_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect::new(updates, imm!(self, 1)))
    }

    pub(super) fn rrx(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rm_val = self.reg_map.get(rm);

        let (sh_val, c) = self.shift_value(rm_val, Some(ArmShift::Rrx))?;

        let n = bd.build_int_compare(IntPredicate::SLT, sh_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, sh_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        let updates = vec![RegUpdate(rd, sh_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect::new(updates, imm!(self, 1)))
    }

    fn lsl_imm(
        &self,
        base: IntValue<'a>,
        imm: u32,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        debug_assert!(imm < 32);
        let bd = &self.builder;
        let one = imm!(self, 1);

        if imm == 0 {
            Ok((base, None))
        } else {
            let shifted = bd.build_left_shift(base, imm!(self, imm), "lsh")?;
            // Get the last bit shifted out
            let rshift = bd.build_right_shift(base, imm!(self, 32 - imm), false, "rsh")?;
            let last_bit = bd.build_and(rshift, one, "b")?;
            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
            Ok((shifted, Some(c)))
        }
    }

    fn lsl_reg(
        &self,
        base: IntValue<'a>,
        reg_id: Reg,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let zero = imm!(self, 0);
        let one = imm!(self, 1);

        let shift_reg = self.reg_map.get(reg_id);
        let curr_c = bd.build_int_cast_sign_flag(self.get_flag(C)?, self.i32_t, false, "c")?;

        // Only use 1st byte of shift reg
        let mask = imm!(self, 0xff);
        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

        let shift_eq_0 = bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
        let shift_lt_32 =
            bd.build_int_compare(IntPredicate::ULT, shift_amt, imm!(self, 32), "slt")?;
        let shift_le_32 =
            bd.build_int_compare(IntPredicate::ULE, shift_amt, imm!(self, 32), "sle")?;
        // Shifter operand calc
        // (reg << shift_reg) if shift_reg < 32 else 0
        let shift_in_range = bd.build_left_shift(base, shift_reg, "sh")?;
        let shift = bd
            .build_select(shift_lt_32, shift_in_range, zero, "shlt")?
            .into_int_value();

        // Carry out calc
        // curr_c if shift_reg = 0 else
        //    (bit(32 - shift_reg) if 0 < shift_reg <= 32) else 0
        let rshift_amt = bd.build_int_sub(imm!(self, 32), shift_amt, "r")?;
        let rshift_in_range = bd.build_right_shift(base, rshift_amt, false, "rsh")?;
        let rshift_eq_0 = bd
            .build_select(shift_eq_0, curr_c, rshift_in_range, "rsheq")?
            .into_int_value();
        let rshift = bd
            .build_select(shift_le_32, rshift_eq_0, zero, "rshle")?
            .into_int_value();

        let last_bit = bd.build_and(rshift, one, "b")?;
        let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "cf")?;
        Ok((shift, Some(c)))
    }

    fn lsr_imm(
        &self,
        base: IntValue<'a>,
        imm: u32,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let zero = imm!(self, 0);
        let one = imm!(self, 1);

        debug_assert!(imm > 0 && imm <= 32);
        if imm == 32 {
            let shift = bd.build_right_shift(base, imm!(self, 31), false, "sh")?;
            let last_bit = bd.build_and(shift, one, "b")?;
            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
            Ok((zero, Some(c)))
        } else {
            let shift = bd.build_right_shift(base, imm!(self, imm), false, "sh")?;
            let shift_1_less = bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
            let last_bit = bd.build_and(shift_1_less, one, "b")?;
            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
            Ok((shift, Some(c)))
        }
    }

    fn lsr_reg(
        &self,
        base: IntValue<'a>,
        reg_id: Reg,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let zero = imm!(self, 0);
        let one = imm!(self, 1);

        let shift_reg = self.reg_map.get(reg_id);
        let curr_c = bd.build_int_cast_sign_flag(self.get_flag(C)?, self.i32_t, false, "c")?;

        // Only use 1st byte of shift reg
        let mask = imm!(self, 0xff);
        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

        let shift_eq_0 = bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
        let shift_lt_32 =
            bd.build_int_compare(IntPredicate::ULT, shift_amt, imm!(self, 32), "slt")?;
        let shift_le_32 =
            bd.build_int_compare(IntPredicate::ULE, shift_amt, imm!(self, 32), "sle")?;
        // Shifter operand calc
        // (reg >> shift_reg) if shift_reg < 32 else 0
        let shift_in_range = bd.build_right_shift(base, shift_reg, false, "sh")?;
        let shift = bd
            .build_select(shift_lt_32, shift_in_range, zero, "shlt")?
            .into_int_value();

        // Carry out calc
        // curr_c if shift_reg = 0 else
        //    (bit(shift_reg - 1) if 0 < shift_reg <= 32) else 0
        let rshift_amt = bd.build_int_sub(shift_amt, one, "r")?;
        let rshift_in_range = bd.build_right_shift(base, rshift_amt, false, "rsh")?;
        let rshift_eq_0 = bd
            .build_select(shift_eq_0, curr_c, rshift_in_range, "rsheq")?
            .into_int_value();
        let rshift = bd
            .build_select(shift_le_32, rshift_eq_0, zero, "rshle")?
            .into_int_value();

        let last_bit = bd.build_and(rshift, one, "b")?;
        let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "cf")?;
        Ok((shift, Some(c)))
    }

    fn asr_imm(
        &self,
        base: IntValue<'a>,
        imm: u32,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        debug_assert!(imm > 0 && imm <= 32);
        let bd = &self.builder;
        let one = imm!(self, 1);

        if imm == 32 {
            let shift = bd.build_right_shift(base, imm!(self, 31), true, "sh")?;
            let last_bit = bd.build_and(shift, one, "b")?;
            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
            Ok((shift, Some(c)))
        } else {
            let shift = bd.build_right_shift(base, imm!(self, imm), true, "sh")?;
            let shift_1_less = bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
            let last_bit = bd.build_and(shift_1_less, one, "b")?;
            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
            Ok((shift, Some(c)))
        }
    }

    fn asr_reg(
        &self,
        base: IntValue<'a>,
        reg_id: Reg,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let zero = imm!(self, 0);
        let one = imm!(self, 1);

        let shift_reg = self.reg_map.get(reg_id);
        let curr_c = self.get_flag(C)?;

        // Only use 1st byte of shift reg
        let mask = imm!(self, 0xff);
        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

        let shift_eq_0 = bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
        let shift_lt_32 =
            bd.build_int_compare(IntPredicate::ULT, shift_amt, imm!(self, 32), "slt")?;
        // Shifter operand calc
        // (reg >> shift_reg) if shift_reg < 32 else 0
        let shift_in_range = bd.build_right_shift(base, shift_amt, true, "sh")?;
        let shift_ge_32 = bd.build_right_shift(base, imm!(self, 31), true, "shge")?;
        let shift = bd
            .build_select(shift_lt_32, shift_in_range, shift_ge_32, "shlt")?
            .into_int_value();

        // Carry out calc
        // curr_c if shift_reg = 0 else
        //    (bit(shift_reg - 1) if 0 < shift_reg <= 32) else 0
        let rshift_amt = bd.build_int_sub(shift_amt, one, "r")?;
        let rshift_in_range = bd.build_right_shift(base, rshift_amt, false, "rsh")?;
        let rshift_31 = bd.build_right_shift(base, imm!(self, 31), false, "rsh31")?;

        let rshift = bd
            .build_select(shift_lt_32, rshift_in_range, rshift_31, "rshle")?
            .into_int_value();

        let last_bit = bd.build_and(rshift, one, "b")?;
        let cf = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "cf")?;
        let c = bd
            .build_select(shift_eq_0, curr_c, cf, "c")?
            .into_int_value();

        Ok((shift, Some(c)))
    }

    fn ror_imm(
        &self,
        base: IntValue<'a>,
        imm: u32,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let one = imm!(self, 1);

        debug_assert!(imm > 0 && imm < 32);
        let rot = call_intrinsic!(bd, self.fshr, base, base, imm!(self, imm)).into_int_value();

        let shift_1_less = bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
        let last_bit = bd.build_and(shift_1_less, one, "b")?;
        let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
        Ok((rot, Some(c)))
    }

    fn ror_reg(
        &self,
        base: IntValue<'a>,
        reg_id: Reg,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let zero = imm!(self, 0);
        let one = imm!(self, 1);

        let shift_reg = self.reg_map.get(reg_id);
        let curr_c = self.get_flag(C)?;

        // Only use 1st byte of shift reg
        let byte_mask = imm!(self, 0xff);
        let shift_amt = bd.build_and(shift_reg, byte_mask, "s")?;

        let byte_eq_0 = bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "beq")?;

        let rot_mask = imm!(self, 0x1f);
        let shift_mod = bd.build_and(shift_reg, rot_mask, "r")?;

        let rot_in_range = call_intrinsic!(bd, self.fshr, base, base, shift_mod).into_int_value();

        let rot = bd
            .build_select(byte_eq_0, base, rot_in_range, "roteq")?
            .into_int_value();

        let shift_mod_1_less = bd.build_int_sub(shift_mod, one, "rl")?;
        let shift_1_less = bd.build_right_shift(base, shift_mod_1_less, false, "shl")?;
        let last_bit = bd.build_and(shift_1_less, one, "b")?;
        let c_non_zero = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "cnz")?;

        let c = bd
            .build_select(byte_eq_0, curr_c, c_non_zero, "c")?
            .into_int_value();

        Ok((rot, Some(c)))
    }

    /// Only has 1 possible immediate value - 1
    fn rrx_imm(&self, base: IntValue<'a>) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = &self.builder;
        let one = imm!(self, 1);

        let curr_c = self.get_flag(C)?;
        let c32 = bd.build_int_cast_sign_flag(curr_c, self.i32_t, false, "c_in")?;
        let rot = call_intrinsic!(bd, self.fshr, c32, base, one).into_int_value();

        let first_bit = bd.build_and(base, one, "lsb")?;
        let c = bd.build_int_compare(IntPredicate::EQ, first_bit, one, "c")?;
        Ok((rot, Some(c)))
    }
}

#[cfg(test)]
mod tests {
    use inkwell::context::Context;

    use super::*;
    use crate::arm::state::ArmState;
    use crate::jit::CompiledFunction;

    /// Shift r0 by the amount in r1 according to the shift type provided and update C flag
    struct ShifterOperandTestCase<'ctx> {
        f: CompiledFunction<'ctx>,
        state: ArmState,
    }

    impl<'ctx> ShifterOperandTestCase<'ctx> {
        fn new(context: &'ctx Context, shift: Option<ArmShift>) -> Self {
            let op = ShifterOperand::Reg {
                reg: Reg::R0,
                shift,
            };
            let mut f = FunctionBuilder::new(context, 0).unwrap();
            f.load_initial_reg_values(&vec![Reg::R0, Reg::R1, Reg::CPSR].into_iter().collect())
                .unwrap();

            let (shifted, carry_out) = f.shifter_operand(op).unwrap();
            f.reg_map.update(Reg::R0, shifted);
            f.reg_map
                .update(Reg::CPSR, f.set_flags(None, None, carry_out, None).unwrap());
            f.write_state_out(&f.reg_map).unwrap();
            f.builder.build_return(None).unwrap();
            let f = f.compile().unwrap();
            Self {
                f,
                state: ArmState::default(),
            }
        }

        fn run(&mut self, r: u32, shift: Option<u32>) -> (u32, bool) {
            self.state.regs[Reg::R0] = r;
            if let Some(shift) = shift {
                self.state.regs[Reg::R1] = shift;
            }
            unsafe {
                self.f.call(&mut self.state);
            }
            (
                self.state.regs[Reg::R0],
                (self.state.regs[Reg::CPSR] & C.0) > 0,
            )
        }
    }

    #[test]
    fn test_shifter_op_lsl_reg() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LslReg(Reg::R1)));

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(12));
        let expect_res = (0b0000_0000_0000_0000_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(0x803e_ff0c));
        let expect_res = (0b0000_0000_0000_0000_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_0001_0000_0000_0000_0000_1101, true); // c unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(7));
        let expect_res = (0b0000_1000_0000_0000_0000_0110_1000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_0001_0000_0000_0000_0000_1101, false); // c unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, Some(31));
        let expect_res = (0b1000_0000_0000_0000_0000_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, Some(32));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1110, Some(32));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, Some(33));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1110, Some(33));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_lsr_reg() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LsrReg(Reg::R1)));

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(16));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_1001_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0x9e01_8410));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_1001_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1010_0000_1001_0000_1110_0000_0000_1101, Some(31));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0001, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, Some(32));
        let expect_res = (0, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0000_1001_0000_1110_0000_0000_1101, Some(32));
        let expect_res = (0, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0110_1111_1001_0000_1110_0000_0000_1101, Some(255));
        let expect_res = (0, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_asr_reg() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::AsrReg(Reg::R1)));

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(8));
        let expect_res = (0b0000_0000_0000_0000_1001_0000_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(4));
        let expect_res = (0b0000_0000_0000_1001_0000_1110_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0));
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, Some(8));
        let expect_res = (0b1111_1111_1000_0000_1001_0000_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, Some(4));
        let expect_res = (0b1111_1000_0000_1001_0000_1110_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(32));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, Some(32));
        let expect_res = (0b1111_1111_1111_1111_1111_1111_1111_1111, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, Some(255));
        let expect_res = (0b1111_1111_1111_1111_1111_1111_1111_1111, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0x100)); // only 1st byte used
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_ror_reg() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::RorReg(Reg::R1)));

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, Some(3));
        let expect_res = (0b1010_1110_0011_0011_1001_1100_0000_1001, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, Some(0));
        let expect_res = (0b0111_0001_1001_1100_1110_0000_0100_1101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1011_0001_1001_1100_1110_0000_0100_1100, Some(31));
        let expect_res = (0b0110_0011_0011_1001_1100_0000_1001_1001, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, Some(0));
        let expect_res = (0b0111_0001_1001_1100_1110_0000_0100_1101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, Some(32));
        let expect_res = (0b1111_0001_1001_1100_1110_0000_0100_1101, true); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, Some(32));
        let expect_res = (0b0111_0001_1001_1100_1110_0000_0100_1101, false); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, Some(163)); // = 3 mod 32
        let expect_res = (0b1011_1110_0011_0011_1001_1100_0000_1001, true);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_lsl_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LslImm(0)));

        tst.state.regs[Reg::CPSR] |= C.0;
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1111_0001_1001_1100_1110_0000_0100_1101, true);
        assert_eq!(res, expect_res);

        tst.state.regs[Reg::CPSR] = 0;
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1111_0001_1001_1100_1110_0000_0100_1101, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LslImm(12)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1100_1110_0000_0100_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LslImm(31)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1100, None);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_lsr_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LsrImm(4)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0000_1111_0001_1001_1100_1110_0000_0100, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_0101, None);
        let expect_res = (0b0000_1111_0001_1001_1100_1110_0000_0100, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::LsrImm(32)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0, true);
        assert_eq!(res, expect_res);
        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_asr_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::AsrImm(4)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1111_1111_0001_1001_1100_1110_0000_0100, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_0101, None);
        let expect_res = (0b0000_0111_0001_1001_1100_1110_0000_0100, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::AsrImm(32)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0xffff_ffff, true);
        assert_eq!(res, expect_res);
        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_ror_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::RorImm(8)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0100_1101_1111_0001_1001_1100_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_1100_1101, None);
        let expect_res = (0b1100_1101_1111_0001_1001_1100_1110_0000, true);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::RorImm(31)));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1110_0011_0011_1001_1100_0000_1001_1011, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1011_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0110_0011_0011_1001_1100_0000_1001_1011, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_rrx() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, Some(ArmShift::Rrx));

        tst.state.regs[Reg::CPSR] |= C.0;
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1111_1000_1100_1110_0111_0000_0010_0110, true);
        assert_eq!(res, expect_res);

        tst.state.regs[Reg::CPSR] = 0;
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0111_1000_1100_1110_0111_0000_0010_0110, true);
        assert_eq!(res, expect_res);
    }

    macro_rules! immediate_shifter_op_tests {
        ($($name:ident: $params:expr,)*) => {
        $(
            #[test]
            // Place the immediate shifter operand value into r0 and update the C flag
            fn $name() {
                let (op, c_set, expected_val, expected_c) = $params;

                let context = Context::create();
                let mut f = FunctionBuilder::new(&context, 0).unwrap();
                f.load_initial_reg_values(&vec![Reg::R0, Reg::R1, Reg::CPSR].into_iter().collect())
                    .unwrap();

                let (shifter, carry_out) = f.shifter_operand(op).unwrap();
                f.reg_map.update(Reg::R0, shifter);
                f.reg_map
                    .update(Reg::CPSR, f.set_flags(None, None, carry_out, None).unwrap());
                f.write_state_out(&f.reg_map).unwrap();
                f.builder.build_return(None).unwrap();
                let f = f.compile().unwrap();

                let mut state = ArmState::default();
                if c_set {
                    state.regs[Reg::CPSR] |= C.0;
                }
                unsafe {
                    f.call(&mut state);
                }
                assert_eq!(state.regs[Reg::R0], expected_val);
                assert_eq!((state.regs[Reg::CPSR] & C.0) > 0, expected_c);
            }
        )*
        };
    }

    immediate_shifter_op_tests! {
        test_unrotated_imm_0: (
            ShifterOperand::Imm{imm: 15, rotate: None}, false,
            15, false
        ),
        test_unrotated_imm_1: (
            ShifterOperand::Imm{imm: 99, rotate: None}, true,
            99, true
        ),
        test_unrotated_imm_2: (
            ShifterOperand::Imm{imm: 0b1001_1100, rotate: None}, false,
            0b1001_1100, false
        ),
        test_unrotated_imm_3: (
            ShifterOperand::Imm{imm: 0b1101_0110, rotate: None}, true,
            0b1101_0110, true
        ),
        test_rotate_imm_0: (
            ShifterOperand::Imm{imm: 1, rotate: Some(30)}, false,
            4, false
        ),
        test_rotate_imm_1: (
            ShifterOperand::Imm{imm: 1, rotate: Some(30)}, true,
            4, false
        ),
        test_rotate_imm_2: (
            ShifterOperand::Imm{imm: 1, rotate: Some(1)}, false,
            2_147_483_648, true
        ),
        test_rotate_imm_3: (
            ShifterOperand::Imm{imm: 1, rotate: Some(1)}, true,
            2_147_483_648, true
        ),
    }
}
