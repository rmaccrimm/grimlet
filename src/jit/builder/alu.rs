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

macro_rules! call_intrinsic {
    ($builder:ident, $self:ident . $intrinsic:ident, $($args:expr),+) => {
        $builder
            .build_call(
                $self.$intrinsic,
                &[$($args.into()),+],
                &format!("{}_res", stringify!(intrinsic))
            )?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| anyhow!("failed to get {} return val", stringify!(intrinsic)))?
    };
}

#[allow(dead_code)]
impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_adc(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::adc)
    }

    pub(super) fn arm_add(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::add)
    }

    pub(super) fn arm_and(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::and)
    }

    pub(super) fn arm_bic(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::bic)
    }

    pub(super) fn arm_cmp(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::cmp)
    }

    pub(super) fn arm_eor(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::eor)
    }

    pub(super) fn arm_mla(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mla)
    }

    pub(super) fn arm_mov(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mov)
    }

    pub(super) fn arm_mvn(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mvn)
    }

    pub(super) fn arm_orr(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::orr)
    }

    pub(super) fn arm_rsb(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::rsb)
    }

    pub(super) fn arm_rsc(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::rsc)
    }

    pub(super) fn arm_sbc(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::sbc)
    }

    pub(super) fn arm_smlal(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::smlal)
    }

    pub(super) fn arm_smull(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::smull)
    }

    pub(super) fn arm_sub(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::sub)
    }

    pub(super) fn arm_teq(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::teq)
    }

    pub(super) fn arm_tst(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::tst)
    }

    pub(super) fn arm_umlal(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::umlal)
    }

    pub(super) fn arm_umull(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::umull)
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
                        debug_assert!(imm < 32);
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
                        debug_assert!(imm > 0 && imm <= 32);
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
                        debug_assert!(imm > 0 && imm <= 32);
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
                        debug_assert!(imm > 0 && imm <= 32);
                        let rot = call_intrinsic!(bd, self.fshr, base, base, imm!(self, imm))
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

                        let rot_in_range =
                            call_intrinsic!(bd, self.fshr, base, base, shift_mod).into_int_value();

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
                        let rot = call_intrinsic!(bd, self.fshr, c32, base, one).into_int_value();

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

    /// TODO - handle rd = pc case
    fn adc(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        let bd = self.builder;
        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;

        let sadd_res_1 =
            call_intrinsic!(bd, self.uadd_with_overflow, rn_val, shifter_op).into_struct_value();
        let s1 = bd
            .build_extract_value(sadd_res_1, 0, "v1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(sadd_res_1, 1, "c1")?
            .into_int_value();

        let sadd_res_2 = call_intrinsic!(bd, self.uadd_with_overflow, s1, c_in).into_struct_value();
        let s2 = bd
            .build_extract_value(sadd_res_2, 0, "v1")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(sadd_res_2, 1, "c2")?
            .into_int_value();

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: s2,
                cpsr: None,
            });
        };

        let uadd_res_1 =
            call_intrinsic!(bd, self.uadd_with_overflow, rn_val, shifter_op).into_struct_value();
        let u1 = bd
            .build_extract_value(uadd_res_1, 0, "v1")?
            .into_int_value();
        let c1 = bd
            .build_extract_value(uadd_res_1, 1, "c1")?
            .into_int_value();

        let uadd_res_2 = call_intrinsic!(bd, self.uadd_with_overflow, u1, c_in).into_struct_value();
        let c2 = bd
            .build_extract_value(uadd_res_2, 1, "c2")?
            .into_int_value();

        let n = bd.build_int_compare(IntPredicate::SLT, s2, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, s2, imm!(self, 0), "z")?;
        let c = bd.build_or(c1, c2, "c")?;
        let v = bd.build_or(v1, v2, "v")?;
        let cpsr = self.set_flags(Some(n), Some(z), Some(c), Some(v))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: s2,
            cpsr: Some(cpsr),
        })
    }

    fn add(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: bd.build_int_add(rn_val, shifter_op, "add")?,
                cpsr: None,
            });
        }

        let sadd_res =
            call_intrinsic!(bd, self.sadd_with_overflow, rn_val, shifter_op).into_struct_value();
        let uadd_res =
            call_intrinsic!(bd, self.uadd_with_overflow, rn_val, shifter_op).into_struct_value();

        let res_val = bd
            .build_extract_value(sadd_res, 0, "res_val")?
            .into_int_value();
        let v = bd.build_extract_value(sadd_res, 1, "v")?.into_int_value();
        let c = bd.build_extract_value(uadd_res, 1, "c")?.into_int_value();

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), Some(c), Some(v))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn and(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        let res_val = bd.build_and(rn_val, shifter_op, "and")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn bic(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        // BIC: Rd = Rn AND NOT(shifter_op)
        let not_shifter = bd.build_not(shifter_op, "not_shift")?;
        let res_val = bd.build_and(rn_val, not_shifter, "bic")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn cmd(&mut self, _instr: ArmInstruction) -> Result<DataProcResult<'a>> { todo!() }

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

    fn eor(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        let res_val = bd.build_xor(rn_val, shifter_op, "eor")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
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

    fn mvn(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let (shifter, c_flag) =
            self.shifter_operand(instr.operands.get(1).expect("Missing operand"))?;

        let res_val = bd.build_not(shifter, "mvn")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn orr(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        let res_val = bd.build_or(rn_val, shifter_op, "orr")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn rsb(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (rm_val, _) = self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        // RSB: Rd = shifter_op - Rn
        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: bd.build_int_sub(rm_val, rn_val, "rsb")?,
                cpsr: None,
            });
        }

        let ures = call_intrinsic!(bd, self.usub_with_overflow, rm_val, rn_val).into_struct_value();
        let sres = call_intrinsic!(bd, self.ssub_with_overflow, rm_val, rn_val).into_struct_value();

        let res_val = bd.build_extract_value(sres, 0, "res_val")?.into_int_value();
        let v_flag = bd.build_extract_value(sres, 1, "v")?.into_int_value();
        let c_flag = bd.build_not(
            bd.build_extract_value(ures, 1, "not_c")?.into_int_value(),
            "c",
        )?;

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), Some(c_flag), Some(v_flag))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

    fn rsc(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) =
            self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        // RSC: Rd = shifter_op - Rn - NOT(C)
        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;
        let not_c = bd.build_not(c_in, "not_c")?;

        let sadd_res_1 =
            call_intrinsic!(bd, self.ssub_with_overflow, shifter_op, rn_val).into_struct_value();
        let s1 = bd
            .build_extract_value(sadd_res_1, 0, "v1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(sadd_res_1, 1, "v1")?
            .into_int_value();

        let sadd_res_2 =
            call_intrinsic!(bd, self.ssub_with_overflow, s1, not_c).into_struct_value();
        let s2 = bd
            .build_extract_value(sadd_res_2, 0, "v2")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(sadd_res_2, 1, "v2")?
            .into_int_value();

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: s2,
                cpsr: None,
            });
        }

        let usub_res_1 =
            call_intrinsic!(bd, self.usub_with_overflow, shifter_op, rn_val).into_struct_value();
        let u1 = bd
            .build_extract_value(usub_res_1, 0, "u1")?
            .into_int_value();
        let c1 = bd
            .build_extract_value(usub_res_1, 1, "c1")?
            .into_int_value();

        let usub_res_2 =
            call_intrinsic!(bd, self.usub_with_overflow, u1, not_c).into_struct_value();
        let c2 = bd
            .build_extract_value(usub_res_2, 1, "c2")?
            .into_int_value();

        let n = bd.build_int_compare(IntPredicate::SLT, s2, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, s2, imm!(self, 0), "z")?;
        let c = bd.build_or(c1, c2, "c")?;
        let v = bd.build_or(v1, v2, "v")?;
        let cpsr = self.set_flags(Some(n), Some(z), Some(c), Some(v))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: s2,
            cpsr: Some(cpsr),
        })
    }

    fn sbc(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (rm_val, _) = self.shifter_operand(instr.operands.get(2).expect("Missing operand"))?;

        // SBC: Rd = Rn - shifter_op - NOT(C)
        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;
        let not_c = bd.build_not(c_in, "not_c")?;

        let sadd_res_1 =
            call_intrinsic!(bd, self.ssub_with_overflow, rn_val, rm_val).into_struct_value();
        let s1 = bd
            .build_extract_value(sadd_res_1, 0, "v1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(sadd_res_1, 1, "v1")?
            .into_int_value();

        let sadd_res_2 =
            call_intrinsic!(bd, self.ssub_with_overflow, s1, not_c).into_struct_value();
        let s2 = bd
            .build_extract_value(sadd_res_2, 0, "v2")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(sadd_res_2, 1, "v2")?
            .into_int_value();

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: s2,
                cpsr: None,
            });
        }

        let usub_res_1 =
            call_intrinsic!(bd, self.usub_with_overflow, rn_val, rm_val).into_struct_value();
        let u1 = bd
            .build_extract_value(usub_res_1, 0, "u1")?
            .into_int_value();
        let c1 = bd
            .build_extract_value(usub_res_1, 1, "c1")?
            .into_int_value();

        let usub_res_2 =
            call_intrinsic!(bd, self.usub_with_overflow, u1, not_c).into_struct_value();
        let c2 = bd
            .build_extract_value(usub_res_2, 1, "c2")?
            .into_int_value();

        let n = bd.build_int_compare(IntPredicate::SLT, s2, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, s2, imm!(self, 0), "z")?;
        let c = bd.build_or(c1, c2, "c")?;
        let v = bd.build_or(v1, v2, "v")?;
        let cpsr = self.set_flags(Some(n), Some(z), Some(c), Some(v))?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: s2,
            cpsr: Some(cpsr),
        })
    }

    fn sub(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (rm_val, _c_flag) =
            self.shifter_operand(instr.operands.get(2).expect("Missing 2nd operand"))?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: bd.build_int_sub(rn_val, rm_val, "sub")?,
                cpsr: None,
            });
        }
        let ures = call_intrinsic!(bd, self.usub_with_overflow, rn_val, rm_val).into_struct_value();
        let sres = call_intrinsic!(bd, self.ssub_with_overflow, rn_val, rm_val).into_struct_value();

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

    fn teq(&mut self, mut instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd because eor expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        let DataProcResult {
            dest: _,
            value,
            cpsr,
        } = self.eor(instr)?;
        Ok(DataProcResult {
            dest: None,
            value,
            cpsr,
        })
    }

    fn tst(&mut self, mut instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd because and expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        let DataProcResult {
            dest: _,
            value,
            cpsr,
        } = self.and(instr)?;
        Ok(DataProcResult {
            dest: None,
            value,
            cpsr,
        })
    }

    fn mla(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rs = instr.get_reg_op(2);
        let rn = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rn_val = self.reg_map.get(rn);

        let mul_res = bd.build_int_mul(rm_val, rs_val, "mul")?;
        let res_val = bd.build_int_add(mul_res, rn_val, "mla")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rd),
                value: res_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            dest: Some(rd),
            value: res_val,
            cpsr: Some(cpsr),
        })
    }

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

    fn smlal(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdhi = instr.get_reg_op(0);
        let rdlo = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        // Combine RdHi:RdLo into 64-bit value
        let rdhi_i64 = bd.build_int_s_extend(rdhi_val, self.llvm_ctx.i64_type(), "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, self.llvm_ctx.i64_type(), "rdlo_i64")?;
        let shift_32 = self.llvm_ctx.i64_type().const_int(32, false);
        let acc = bd.build_left_shift(rdhi_i64, shift_32, "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        // Sign-extend operands to i64 and multiply
        let rm_i64 = bd.build_int_s_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        // Add to accumulator
        let res = bd.build_int_add(mul_res, acc, "smlal")?;

        // Extract high 32 bits for RdHi
        let hi_val = bd.build_right_shift(res, shift_32, true, "hi")?;
        let hi_val = bd.build_int_truncate(hi_val, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rdhi),
                value: hi_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(
            IntPredicate::SLT,
            res,
            self.llvm_ctx.i64_type().const_zero(),
            "n",
        )?;
        let z = bd.build_int_compare(
            IntPredicate::EQ,
            res,
            self.llvm_ctx.i64_type().const_zero(),
            "z",
        )?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            dest: Some(rdhi),
            value: hi_val,
            cpsr: Some(cpsr),
        })
    }

    fn smull(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdhi = instr.get_reg_op(0);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Sign-extend to i64 and multiply
        let rm_i64 = bd.build_int_s_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "smull")?;

        // Note: RdLo would be updated separately in actual instruction dispatcher
        let _lo_val = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;

        // Extract high 32 bits for RdHi
        let shift_32 = self.llvm_ctx.i64_type().const_int(32, false);
        let hi_val = bd.build_right_shift(mul_res, shift_32, true, "hi")?;
        let hi_val = bd.build_int_truncate(hi_val, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rdhi), // We'll update both in the caller
                value: hi_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(
            IntPredicate::SLT,
            mul_res,
            self.llvm_ctx.i64_type().const_zero(),
            "n",
        )?;
        let z = bd.build_int_compare(
            IntPredicate::EQ,
            mul_res,
            self.llvm_ctx.i64_type().const_zero(),
            "z",
        )?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            dest: Some(rdhi),
            value: hi_val,
            cpsr: Some(cpsr),
        })
    }

    fn umlal(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdhi = instr.get_reg_op(0);
        let rdlo = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        // Combine RdHi:RdLo into 64-bit value
        let rdhi_i64 = bd.build_int_z_extend(rdhi_val, self.llvm_ctx.i64_type(), "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, self.llvm_ctx.i64_type(), "rdlo_i64")?;
        let shift_32 = self.llvm_ctx.i64_type().const_int(32, false);
        let acc = bd.build_left_shift(rdhi_i64, shift_32, "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        // Zero-extend operands to i64 and multiply
        let rm_i64 = bd.build_int_z_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        // Add to accumulator
        let res = bd.build_int_add(mul_res, acc, "umlal")?;

        // Extract high 32 bits for RdHi
        let hi_val = bd.build_right_shift(res, shift_32, false, "hi")?;
        let hi_val = bd.build_int_truncate(hi_val, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rdhi),
                value: hi_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(
            IntPredicate::SLT,
            res,
            self.llvm_ctx.i64_type().const_zero(),
            "n",
        )?;
        let z = bd.build_int_compare(
            IntPredicate::EQ,
            res,
            self.llvm_ctx.i64_type().const_zero(),
            "z",
        )?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            dest: Some(rdhi),
            value: hi_val,
            cpsr: Some(cpsr),
        })
    }

    fn umull(&mut self, instr: ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdhi = instr.get_reg_op(0);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Zero-extend to i64 and multiply
        let rm_i64 = bd.build_int_z_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "umull")?;

        // Note: RdLo would be updated separately in actual instruction dispatcher
        let _lo_val = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;

        // Extract high 32 bits for RdHi
        let shift_32 = self.llvm_ctx.i64_type().const_int(32, false);
        let hi_val = bd.build_right_shift(mul_res, shift_32, false, "hi")?;
        let hi_val = bd.build_int_truncate(hi_val, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                dest: Some(rdhi), // We'll update both in the caller
                value: hi_val,
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(
            IntPredicate::SLT,
            mul_res,
            self.llvm_ctx.i64_type().const_zero(),
            "n",
        )?;
        let z = bd.build_int_compare(
            IntPredicate::EQ,
            mul_res,
            self.llvm_ctx.i64_type().const_zero(),
            "z",
        )?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            dest: Some(rdhi),
            value: hi_val,
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

    /// Shift r0 by the amount in r1 according to the shift type provided and update C flag
    struct ShifterOperandTestCase<'ctx> {
        f: CompiledFunction<'ctx>,
        state: ArmState,
    }

    impl<'ctx> ShifterOperandTestCase<'ctx> {
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
            f.reg_map.update(Reg::R0, shifted);
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
        let mut tst =
            ShifterOperandTestCase::new(&ctx, ArmShift::LslReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(12));
        let expect_res = (0b0000_0000_0000_0000_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_0001_0000_0000_0000_0000_1101, Some(0x803eff0c));
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
        let mut tst =
            ShifterOperandTestCase::new(&ctx, ArmShift::LsrReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(16));
        let expect_res = (0b0000_0000_0000_0000_0000_0000_1001_0000, true);
        assert_eq!(res, expect_res);

        // only last byte used
        let res = tst.run(0b0000_0000_1001_0000_1110_0000_0000_1101, Some(0x9e018410));
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
        let mut tst =
            ShifterOperandTestCase::new(&ctx, ArmShift::AsrReg(RegId(ArmReg::ARM_REG_R1 as u16)));

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
        let mut tst =
            ShifterOperandTestCase::new(&ctx, ArmShift::RorReg(RegId(ArmReg::ARM_REG_R1 as u16)));

        let res = tst.run(0b01110001100111001110000001001101, Some(3));
        let expect_res = (0b10101110001100111001110000001001, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, Some(0));
        let expect_res = (0b01110001100111001110000001001101, true); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b1011_0001_1001_1100_1110_0000_0100_1100, Some(31));
        let expect_res = (0b0110_0011_0011_1001_1100_0000_1001_1001, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, Some(0));
        let expect_res = (0b01110001100111001110000001001101, false); // unchanged
        assert_eq!(res, expect_res);

        let res = tst.run(0b11110001100111001110000001001101, Some(32));
        let expect_res = (0b11110001100111001110000001001101, true); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b01110001100111001110000001001101, Some(32));
        let expect_res = (0b01110001100111001110000001001101, false); // bit 31
        assert_eq!(res, expect_res);

        let res = tst.run(0b11110001100111001110000001001101, Some(163)); // = 3 mod 32
        let expect_res = (0b10111110001100111001110000001001, true);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_lsl_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Lsl(0));

        tst.state.regs[Reg::CPSR] |= C.0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b11110001100111001110000001001101, true);
        assert_eq!(res, expect_res);

        tst.state.regs[Reg::CPSR] = 0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b11110001100111001110000001001101, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Lsl(12));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1100_1110_0000_0100_1101_0000_0000_0000, true);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Lsl(31));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1100, None);
        let expect_res = (0b0000_0000_0000_0000_0000_0000_0000_0000, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_lsr_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Lsr(4));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0000_1111_0001_1001_1100_1110_0000_0100, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_0101, None);
        let expect_res = (0b0000_1111_0001_1001_1100_1110_0000_0100, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Lsr(32));
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
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Asr(4));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b1111_1111_0001_1001_1100_1110_0000_0100, true);
        assert_eq!(res, expect_res);

        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_0101, None);
        let expect_res = (0b0000_0111_0001_1001_1100_1110_0000_0100, false);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Asr(32));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0xffffffff, true);
        assert_eq!(res, expect_res);
        let res = tst.run(0b0111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0, false);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_shifter_op_ror_imm() {
        let ctx = Context::create();
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Ror(8));
        let res = tst.run(0b1111_0001_1001_1100_1110_0000_0100_1101, None);
        let expect_res = (0b0100_1101_1111_0001_1001_1100_1110_0000, false);
        assert_eq!(res, expect_res);

        let res = tst.run(0b1111_0001_1001_1100_1110_0000_1100_1101, None);
        let expect_res = (0b1100_1101_1111_0001_1001_1100_1110_0000, true);
        assert_eq!(res, expect_res);

        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Ror(31));
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
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Rrx(13)); // imm value is ignored

        tst.state.regs[Reg::CPSR] |= C.0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b11111000110011100111000000100110, true);
        assert_eq!(res, expect_res);

        tst.state.regs[Reg::CPSR] = 0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b01111000110011100111000000100110, true);
        assert_eq!(res, expect_res);
    }

    #[test]
    fn test_adc() { todo!() }

    #[test]
    fn test_add() { todo!() }
}
