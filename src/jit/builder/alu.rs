use anyhow::{Context as _, Result, anyhow};
use capstone::arch::arm::{ArmCC, ArmOperand, ArmOperandType, ArmShift};
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::arm::disasm::ArmInstruction;
use crate::jit::FunctionBuilder;
use crate::jit::builder::flags::C;

struct DataProcResult<'a> {
    dest: Option<Reg>,
    value: IntValue<'a>,
    cpsr: Option<IntValue<'a>>,
}

macro_rules! exec_and_expect {
    ($self:ident, $arg:ident, Self::$method:ident) => {
        $self
            .exec_alu_conditional($arg, Self::$method)
            .with_context(|| format!("\"{}\"", stringify!($method)))
            .expect("LLVM codegen failed")
    };
}

macro_rules! imm {
    ($self:ident, $i:expr) => {
        $self.i32_t.const_int($i as u64, false)
    };
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::cmp)
    }

    pub(super) fn arm_mov(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mov)
    }

    pub(super) fn arm_sub(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::sub)
    }

    pub(super) fn arm_mul(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mul)
    }

    /// Wraps a function for emitting an instruction in a conditional block, evaluates flags and
    /// executes based on instruction condition Leaves the builder positioned in the else block and
    /// emits code to increment program counter.
    fn exec_alu_conditional<F>(&mut self, instr: ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<DataProcResult<'a>>,
    {
        let mode = instr.mode;

        if instr.cond == ArmCC::ARM_CC_AL {
            let calc_result = inner(self, instr)?;
            if let Some(rd) = calc_result.dest {
                self.reg_map.update(rd, calc_result.value);
            }
            if let Some(cpsr) = calc_result.cpsr {
                self.reg_map.update(Reg::CPSR, cpsr);
            }

            self.increment_pc(mode);
            return Ok(());
        }

        let ctx = self.llvm_ctx;
        let bd = self.builder;
        let if_block = ctx.append_basic_block(self.func, "if");
        let end_block = ctx.append_basic_block(self.func, "end");
        // If not executing instruction, init dest value is kept;
        let rd = instr.get_reg_op(0);
        let dest_init = self.reg_map.get(rd);
        let cpsr_init = self.reg_map.cpsr();
        let cond = self.eval_cond(instr.cond)?;
        bd.build_conditional_branch(cond, if_block, end_block)?;
        bd.position_at_end(if_block);

        let calc_result = inner(self, instr)?;
        bd.build_unconditional_branch(end_block)?;
        bd.position_at_end(end_block);

        if let Some(rd) = calc_result.dest {
            let phi = bd.build_phi(self.i32_t, "phi_dest")?;
            phi.add_incoming(&[
                (&dest_init, self.current_block),
                (&calc_result.value, if_block),
            ]);
            self.reg_map
                .update(rd, phi.as_basic_value().into_int_value());
        }
        if let Some(cpsr) = calc_result.cpsr {
            let phi = bd.build_phi(self.i32_t, "phi_cpsr")?;
            phi.add_incoming(&[(&cpsr_init, self.current_block), (&cpsr, if_block)]);
            self.reg_map
                .update(Reg::CPSR, phi.as_basic_value().into_int_value());
        }
        self.increment_pc(mode);
        self.current_block = end_block;
        Ok(())
    }

    // Returns (i32, Option<i1>) - the value of the operand and the shifter carry-out value (None if
    // unaffected)
    fn shifter_operand(
        &self,
        operand: &ArmOperand,
    ) -> Result<(IntValue<'a>, Option<IntValue<'a>>)> {
        let bd = self.builder;
        let one = imm!(self, 1);
        let zero = imm!(self, 0);

        match operand.op_type {
            ArmOperandType::Imm(imm) => {
                let imm_val = self.i32_t.const_int(imm as u64, false);
                let shifted = bd.build_right_shift(imm_val, imm!(self, 31), false, "sh")?;
                let imm_msb = bd.build_int_compare(IntPredicate::EQ, shifted, one, "imm_msb")?;
                Ok((imm_val, Some(imm_msb)))
            }
            ArmOperandType::Reg(reg_id) => {
                let base = self.reg_map.get(Reg::from(reg_id));

                match operand.shift {
                    ArmShift::Invalid => Ok((base, None)),
                    ArmShift::Lsl(imm) => {
                        // imm guaranteed to be < 32
                        if imm == 0 {
                            Ok((base, None))
                        } else {
                            let shifted = bd.build_left_shift(base, imm!(self, imm), "lsh")?;
                            // Get the last bit shifted out
                            let rshift =
                                bd.build_right_shift(base, imm!(self, 32 - imm), false, "rsh")?;
                            let last_bit = bd.build_and(rshift, one, "b")?;
                            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                            Ok((shifted, Some(c)))
                        }
                    }
                    ArmShift::LslReg(reg_id) => {
                        let shift_reg = self.reg_map.get(Reg::from(reg_id));
                        let curr_c =
                            bd.build_int_cast_sign_flag(self.get_flag(C)?, self.i32_t, false, "c")?;

                        // Only use 1st byte of shift reg
                        let mask = imm!(self, 0xff);
                        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

                        let shift_eq_0 =
                            bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
                        let shift_lt_32 = bd.build_int_compare(
                            IntPredicate::ULT,
                            shift_amt,
                            imm!(self, 32),
                            "slt",
                        )?;
                        let shift_le_32 = bd.build_int_compare(
                            IntPredicate::ULE,
                            shift_amt,
                            imm!(self, 32),
                            "sle",
                        )?;
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
                        let rshift_in_range =
                            bd.build_right_shift(base, rshift_amt, false, "rsh")?;
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
                    ArmShift::Lsr(imm) => {
                        // imm guaranteed to be between 1 and 32
                        if imm == 32 {
                            let shift = bd.build_right_shift(base, imm!(self, 31), false, "sh")?;
                            let last_bit = bd.build_and(shift, one, "b")?;
                            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                            Ok((zero, Some(c)))
                        } else {
                            let shift = bd.build_right_shift(base, imm!(self, imm), false, "sh")?;
                            let shift_1_less =
                                bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
                            let last_bit = bd.build_and(shift_1_less, one, "b")?;
                            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                            Ok((shift, Some(c)))
                        }
                    }
                    ArmShift::LsrReg(reg_id) => {
                        let shift_reg = self.reg_map.get(Reg::from(reg_id));
                        let curr_c =
                            bd.build_int_cast_sign_flag(self.get_flag(C)?, self.i32_t, false, "c")?;

                        // Only use 1st byte of shift reg
                        let mask = imm!(self, 0xff);
                        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

                        let shift_eq_0 =
                            bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
                        let shift_lt_32 = bd.build_int_compare(
                            IntPredicate::ULT,
                            shift_amt,
                            imm!(self, 32),
                            "slt",
                        )?;
                        let shift_le_32 = bd.build_int_compare(
                            IntPredicate::ULE,
                            shift_amt,
                            imm!(self, 32),
                            "sle",
                        )?;
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
                        let rshift_in_range =
                            bd.build_right_shift(base, rshift_amt, false, "rsh")?;
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
                    ArmShift::Asr(imm) => {
                        if imm == 32 {
                            let shift = bd.build_right_shift(base, imm!(self, 31), true, "sh")?;
                            let last_bit = bd.build_and(shift, one, "b")?;
                            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                            Ok((shift, Some(c)))
                        } else {
                            let shift = bd.build_right_shift(base, imm!(self, imm), true, "sh")?;
                            let shift_1_less =
                                bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
                            let last_bit = bd.build_and(shift_1_less, one, "b")?;
                            let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                            Ok((shift, Some(c)))
                        }
                    }
                    ArmShift::AsrReg(reg_id) => {
                        let shift_reg = self.reg_map.get(Reg::from(reg_id));
                        let curr_c = self.get_flag(C)?;

                        // Only use 1st byte of shift reg
                        let mask = imm!(self, 0xff);
                        let shift_amt = bd.build_and(shift_reg, mask, "s")?;

                        let shift_eq_0 =
                            bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "seq")?;
                        let shift_lt_32 = bd.build_int_compare(
                            IntPredicate::ULT,
                            shift_amt,
                            imm!(self, 32),
                            "slt",
                        )?;
                        // Shifter operand calc
                        // (reg >> shift_reg) if shift_reg < 32 else 0
                        let shift_in_range = bd.build_right_shift(base, shift_amt, true, "sh")?;
                        let shift_ge_32 =
                            bd.build_right_shift(base, imm!(self, 31), true, "shge")?;
                        let shift = bd
                            .build_select(shift_lt_32, shift_in_range, shift_ge_32, "shlt")?
                            .into_int_value();

                        // Carry out calc
                        // curr_c if shift_reg = 0 else
                        //    (bit(shift_reg - 1) if 0 < shift_reg <= 32) else 0
                        let rshift_amt = bd.build_int_sub(shift_amt, one, "r")?;
                        let rshift_in_range =
                            bd.build_right_shift(base, rshift_amt, false, "rsh")?;
                        let rshift_31 =
                            bd.build_right_shift(base, imm!(self, 31), false, "rsh31")?;

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
                    ArmShift::Ror(imm) => {
                        // imm guarantedd to be between 1 and 32 (0 encodes rrx)
                        let rot = bd
                            .build_call(
                                self.fshr,
                                &[base.into(), base.into(), imm!(self, imm).into()],
                                "rot",
                            )?
                            .try_as_basic_value()
                            .left()
                            .ok_or_else(|| anyhow!("failed to get fshr return val"))?
                            .into_int_value();

                        let shift_1_less =
                            bd.build_right_shift(base, imm!(self, imm - 1), false, "shl")?;
                        let last_bit = bd.build_and(shift_1_less, one, "b")?;
                        let c = bd.build_int_compare(IntPredicate::EQ, last_bit, one, "c")?;
                        Ok((rot, Some(c)))
                    }
                    ArmShift::RorReg(reg_id) => {
                        let shift_reg = self.reg_map.get(Reg::from(reg_id));
                        let curr_c = self.get_flag(C)?;

                        // Only use 1st byte of shift reg
                        let byte_mask = imm!(self, 0xff);
                        let shift_amt = bd.build_and(shift_reg, byte_mask, "s")?;

                        let byte_eq_0 =
                            bd.build_int_compare(IntPredicate::EQ, shift_amt, zero, "beq")?;

                        let rot_mask = imm!(self, 0x1f);
                        let shift_mod = bd.build_and(shift_reg, rot_mask, "r")?;

                        let rot_in_range = bd
                            .build_call(
                                self.fshr,
                                &[base.into(), base.into(), shift_mod.into()],
                                "rot",
                            )?
                            .try_as_basic_value()
                            .left()
                            .ok_or_else(|| anyhow!("failed to get fshr return val"))?
                            .into_int_value();

                        let rot = bd
                            .build_select(byte_eq_0, base, rot_in_range, "roteq")?
                            .into_int_value();

                        let shift_mod_1_less = bd.build_int_sub(shift_mod, one, "rl")?;
                        let shift_1_less =
                            bd.build_right_shift(base, shift_mod_1_less, false, "shl")?;
                        let last_bit = bd.build_and(shift_1_less, one, "b")?;
                        let c_non_zero =
                            bd.build_int_compare(IntPredicate::EQ, last_bit, one, "cnz")?;

                        let c = bd
                            .build_select(byte_eq_0, curr_c, c_non_zero, "c")?
                            .into_int_value();

                        Ok((rot, Some(c)))
                    }
                    ArmShift::Rrx(_) => {
                        let curr_c = self.get_flag(C)?;
                        let c32 = bd.build_int_cast_sign_flag(curr_c, self.i32_t, false, "c_in")?;
                        let rot = bd
                            .build_call(self.fshr, &[c32.into(), base.into()], "rot")?
                            .try_as_basic_value()
                            .left()
                            .ok_or_else(|| anyhow!("failed to get fshr return val"))?
                            .into_int_value();

                        let first_bit = bd.build_and(base, one, "lsb")?;
                        let c = bd.build_int_compare(IntPredicate::EQ, first_bit, one, "c")?;
                        Ok((rot, Some(c)))
                    }
                    ArmShift::RrxReg(_reg_id) => panic!("unsupported operand (RRX reg)"),
                }
            }
            ArmOperandType::Mem(_) => todo!(),
            _ => panic!("unhandled operand type"),
        }
    }

    fn mov(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd = instr.get_reg_op(0);
        let (shifter, c) =
            self.shifter_operand(instr.operands.get(1).expect("missing 2nd operand"))?;

        if instr.updates_flags {
            let bd = self.builder;
            let zero = self.i32_t.const_int(0, false);
            let n = bd.build_int_compare(IntPredicate::SLT, shifter, zero, "n")?;
            let z = bd.build_int_compare(IntPredicate::EQ, shifter, zero, "z")?;
            let cpsr = self.set_flags(Some(n), Some(z), c, None)?;
            Ok(DataProcResult {
                dest: Some(rd),
                value: shifter,
                cpsr: Some(cpsr),
            })
        } else {
            Ok(DataProcResult {
                dest: Some(rd),
                value: shifter,
                cpsr: None,
            })
        }
    }

    // Flags not implemented
    fn mul(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rs = instr.get_reg_op(2);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let res = self.builder.build_int_mul(rm_val, rs_val, "mul")?;
        Ok(DataProcResult {
            dest: Some(rd),
            value: res,
            cpsr: None,
        })
    }

    fn cmp(&mut self, mut instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let op1 = instr.operands[0].clone();
        let op2 = instr.operands[1].clone();
        // Insert a dummy operand because sub expects it.
        instr.operands = vec![op1.clone(), op1, op2];
        let DataProcResult {
            dest: _,
            value,
            cpsr,
        } = self.sub(instr)?;
        Ok(DataProcResult {
            dest: None,
            value,
            cpsr,
        })
    }

    // TODO some way of sharing shifter operand code
    fn sub(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let rm_val = match instr.operands.get(2).expect("Missing 2nd operand").op_type {
            ArmOperandType::Reg(reg_id) => {
                let rm = Reg::from(reg_id);
                self.reg_map.get(rm)
            }
            ArmOperandType::Imm(imm) => self.i32_t.const_int(imm as u64, false),
            _ => panic!("unhandled op_type"),
        };

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: bd.build_int_sub(rn_val, rm_val, "sub")?,
                cpsr: None,
            });
        }
        let ures = bd
            .build_call(
                self.usub_with_overflow,
                &[rn_val.into(), rm_val.into()],
                "ures",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_struct_value();

        let sres = bd
            .build_call(
                self.ssub_with_overflow,
                &[rn_val.into(), rm_val.into()],
                "sres",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_struct_value();

        let sres_val = bd
            .build_extract_value(sres, 0, "sres_val")?
            .into_int_value();

        let v_flag = bd.build_extract_value(sres, 1, "v")?.into_int_value();
        let c_flag = bd.build_not(
            bd.build_extract_value(ures, 1, "not_c")?.into_int_value(),
            "c",
        )?;
        let z_flag =
            bd.build_int_compare(IntPredicate::EQ, sres_val, self.i32_t.const_zero(), "n")?;
        let n_flag =
            bd.build_int_compare(IntPredicate::SLT, sres_val, self.i32_t.const_zero(), "z")?;

        let cpsr = self.set_flags(Some(n_flag), Some(z_flag), Some(c_flag), Some(v_flag))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: sres_val,
            cpsr: Some(cpsr),
        })
    }
}

#[cfg(test)]
mod tests {

    use capstone::RegId;
    use capstone::arch::arm::ArmReg;
    use inkwell::context::Context;

    use super::*;
    use crate::arm::cpu::ArmState;
    use crate::jit::{CompiledFunction, Compiler};

    struct ShiftTestCase<'ctx> {
        f: CompiledFunction<'ctx>,
        state: ArmState,
    }

    impl<'ctx> ShiftTestCase<'ctx> {
        fn new(context: &'ctx Context, shift: ArmShift) -> Self {
            let op = ArmOperand {
                op_type: ArmOperandType::Reg(RegId(ArmReg::ARM_REG_R0 as u16)),
                shift,
                ..Default::default()
            };
            let mut compiler = Compiler::new(context);
            let mut f = compiler.new_function(0, None);
            f.load_initial_reg_values(&vec![Reg::R0, Reg::R1, Reg::CPSR].into_iter().collect())
                .unwrap();

            let (shifted, carry_out) = f.shifter_operand(&op).unwrap();
            let carry32 = f
                .builder
                .build_int_cast_sign_flag(carry_out.unwrap(), f.i32_t, false, "c32")
                .unwrap();
            f.reg_map.update(Reg::R0, shifted);
            f.reg_map.update(Reg::R1, carry32);
            f.reg_map
                .update(Reg::CPSR, f.set_flags(None, None, carry_out, None).unwrap());
            f.write_state_out().unwrap();
            f.builder.build_return(None).unwrap();
            let f = f.compile().unwrap();
            Self {
                f,
                state: ArmState::default(),
            }
        }

        fn run(&mut self, r: u32, shift: u32) -> (u32, bool) {
            self.state.regs[Reg::R0] = r;
            self.state.regs[Reg::R1] = shift;
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
    fn test_lsl_reg() {
        let ctx = Context::create();
        let mut tst = ShiftTestCase::new(&ctx, ArmShift::LslReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, 12);
        let expect_res = (0b0000_0000_0000_0000_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, 0x803eff0c);
        let expect_res = (0b0000_0000_0000_0000_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_0001_0000_0000_0000_0000_1101, true); // c unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, 7);
        let expect_res = (0b0000_1000_0000_0000_0000_0110_1000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_0001_0000_0000_0000_0000_1101, false); // c unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, 31);
        let expect_res = (0b1000_0000_0000_0000_0000_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, 32);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1110, 32);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1111, 33);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1110, 33);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_lsr_reg() {
        let ctx = Context::create();
        let mut tst = ShiftTestCase::new(&ctx, ArmShift::LsrReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 16);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_1001_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0x9e018410);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_1001_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1010_0000_1001_0000_1110_0000_0000_1101, 31);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0001, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, 32);
        let expect_res = (0, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0000_1001_0000_1110_0000_0000_1101, 32);
        let expect_res = (0, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0110_1111_1001_0000_1110_0000_0000_1101, 255);
        let expect_res = (0, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_asr_reg() {
        let ctx = Context::create();
        let mut tst = ShiftTestCase::new(&ctx, ArmShift::AsrReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 8);
        let expect_res = (0b0000_0000_0000_0000_1001_0000_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 4);
        let expect_res = (0b0000_0000_0000_1001_0000_1110_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0);
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, 8);
        let expect_res = (0b1111_1111_1000_0000_1001_0000_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, 4);
        let expect_res = (0b1111_1000_0000_1001_0000_1110_0000_0000, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 32);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, 32);
        let expect_res = (0b1111_1111_1111_1111_1111_1111_1111_1111, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1000_0000_1001_0000_1110_0000_0000_1101, 255);
        let expect_res = (0b1111_1111_1111_1111_1111_1111_1111_1111, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, 0x100); // only 1st byte used
        let expect_res = (0b0000_0000_1001_0000_1110_0000_0000_1101, true); // unchanged
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_ror_reg() {
        let ctx = Context::create();
        let mut tst = ShiftTestCase::new(&ctx, ArmShift::RorReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b01110001100111001110000001001101, 3);
        let expect_res = (0b10101110001100111001110000001001, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, 0);
        let expect_res = (0b01110001100111001110000001001101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1011_0001_1001_1100_1110_0000_0100_1100, 31);
        let expect_res = (0b0110_0011_0011_1001_1100_0000_1001_1001, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, 0);
        let expect_res = (0b01110001100111001110000001001101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b11110001100111001110000001001101, 32);
        let expect_res = (0b11110001100111001110000001001101, true); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, 32);
        let expect_res = (0b01110001100111001110000001001101, false); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b11110001100111001110000001001101, 163); // = 3 mod 32
        let expect_res = (0b10111110001100111001110000001001, true);
        assert_eq!(res, expect_res);
    }
}
