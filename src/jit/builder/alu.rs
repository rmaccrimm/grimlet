use anyhow::{Context as _, Result, anyhow};
use capstone::RegId;
use capstone::arch::arm::{ArmOperand, ArmOperandType, ArmShift};
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::Reg;
use crate::jit::FunctionBuilder;
use crate::jit::builder::flags::C;

// A register and the value to write to it
pub(super) struct RegUpdate<'a> {
    pub reg: Reg,
    pub value: IntValue<'a>,
}

// Indicates what should be done with the result of a calculation
enum DataProcAction<'a> {
    Ignored,
    SingleUpdate(RegUpdate<'a>),
    DoubleUpdate((RegUpdate<'a>, RegUpdate<'a>)),
}

impl<'a> DataProcAction<'a> {
    fn single(reg: Reg, value: IntValue<'a>) -> Self {
        Self::SingleUpdate(RegUpdate { reg, value })
    }

    fn double(r1: Reg, v1: IntValue<'a>, r2: Reg, v2: IntValue<'a>) -> Self {
        Self::DoubleUpdate((
            RegUpdate { reg: r1, value: v1 },
            RegUpdate { reg: r2, value: v2 },
        ))
    }
}

// Return type for all data processing operation builder methods
struct DataProcResult<'a> {
    action: DataProcAction<'a>,
    cpsr: Option<IntValue<'a>>,
}

macro_rules! exec_and_expect {
    ($self:ident, $arg:ident, Self::$method:ident) => {
        $self
            .exec_alu_conditional(&$arg, Self::$method)
            .with_context(|| format!("{:?}", $arg))
            .expect("LLVM codegen failed")
    };
}

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

    pub(super) fn arm_cmn(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::cmn)
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

    pub(super) fn arm_mrs(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::mrs)
    }

    pub(super) fn arm_msr(&mut self, instr: ArmInstruction) {
        exec_and_expect!(self, instr, Self::msr)
    }

    /// Wraps a function for emitting an instruction in a conditional block, evaluates flags and
    /// executes based on instruction condition Leaves the builder positioned in the else block and
    /// emits code to increment program counter.
    fn exec_alu_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, &ArmInstruction) -> Result<DataProcResult<'a>>,
    {
        let mode = instr.mode;
        let ctx = self.llvm_ctx;
        let bd = self.builder;
        let if_block = ctx.append_basic_block(self.func, "if");
        let end_block = ctx.append_basic_block(self.func, "end");

        // Potentially need phi nodes for up to 2 dest registers
        let rd_0_init = if let Some(op) = instr.operands.first()
            && let ArmOperandType::Reg(reg_id) = op.op_type
        {
            Some(self.reg_map.get(Reg::from(reg_id)))
        } else {
            None
        };

        let rd_1_init = if let Some(op) = instr.operands.get(1)
            && let ArmOperandType::Reg(reg_id) = op.op_type
        {
            Some(self.reg_map.get(Reg::from(reg_id)))
        } else {
            None
        };

        let cpsr_init = self.reg_map.cpsr();
        let cond = self.eval_cond(instr.cond)?;
        bd.build_conditional_branch(cond, if_block, end_block)?;
        bd.position_at_end(if_block);

        let calc_result = inner(self, instr)?;
        bd.build_unconditional_branch(end_block)?;
        bd.position_at_end(end_block);

        // Update the output register(s) with phi values depending on branch taken
        match calc_result.action {
            DataProcAction::Ignored => {}
            DataProcAction::SingleUpdate(RegUpdate { reg, value }) => {
                let phi = bd.build_phi(self.i32_t, "phi")?;
                phi.add_incoming(&[
                    (
                        &rd_0_init.expect("more results than register operands"),
                        self.current_block,
                    ),
                    (&value, if_block),
                ]);
                self.reg_map
                    .update(reg, phi.as_basic_value().into_int_value());
            }
            DataProcAction::DoubleUpdate((
                RegUpdate { reg: r0, value: v0 },
                RegUpdate { reg: r1, value: v1 },
            )) => {
                let phi_0 = bd.build_phi(self.i32_t, "phi_0")?;
                let phi_1 = bd.build_phi(self.i32_t, "phi_1")?;
                phi_0.add_incoming(&[
                    (
                        &rd_0_init.expect("more results than register operands"),
                        self.current_block,
                    ),
                    (&v0, if_block),
                ]);
                phi_1.add_incoming(&[
                    (
                        &rd_1_init.expect("more results than register operands"),
                        self.current_block,
                    ),
                    (&v1, if_block),
                ]);
                self.reg_map
                    .update(r0, phi_0.as_basic_value().into_int_value());
                self.reg_map
                    .update(r1, phi_1.as_basic_value().into_int_value());
            }
        }

        // Update CPSR if instruction sets flags
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
                        debug_assert!(imm > 0 && imm < 32);
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
                    ArmShift::Rrx(imm) => {
                        debug_assert_eq!(imm, 1);
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
    fn adc(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        let bd = self.builder;
        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;

        if !instr.updates_flags {
            let add_1 = bd.build_int_add(rn_val, shifter_op, "add_1")?;
            let add_2 = bd.build_int_add(add_1, c_in, "add_2")?;
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, add_2),
                cpsr: None,
            });
        };

        let sadd_res_1 =
            call_intrinsic!(bd, self.uadd_with_overflow, rn_val, shifter_op).into_struct_value();
        let s1 = bd
            .build_extract_value(sadd_res_1, 0, "sres1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(sadd_res_1, 1, "c1")?
            .into_int_value();

        let sadd_res_2 = call_intrinsic!(bd, self.uadd_with_overflow, s1, c_in).into_struct_value();
        let s2 = bd
            .build_extract_value(sadd_res_2, 0, "sres2")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(sadd_res_2, 1, "c2")?
            .into_int_value();

        let uadd_res_1 =
            call_intrinsic!(bd, self.uadd_with_overflow, rn_val, shifter_op).into_struct_value();
        let u1 = bd
            .build_extract_value(uadd_res_1, 0, "ures")?
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
            action: DataProcAction::single(rd, s2),
            cpsr: Some(cpsr),
        })
    }

    /// TODO rd = pc case
    fn add(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(
                    rd,
                    bd.build_int_add(rn_val, shifter_op, "add_res")?,
                ),
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
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn and(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(&instr.operands[2])?;

        let res_val = bd.build_and(rn_val, shifter_op, "and")?;
        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }
        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn bic(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(&instr.operands[2])?;

        let not_shifter = bd.build_not(shifter_op, "not_shift")?;
        let res_val = bd.build_and(rn_val, not_shifter, "bic")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn cmn(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        let DataProcResult { action: _, cpsr } = self.add(&instr)?;
        Ok(DataProcResult {
            action: DataProcAction::Ignored,
            cpsr,
        })
    }

    fn cmp(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        let DataProcResult { action: _, cpsr } = self.sub(&instr)?;
        Ok(DataProcResult {
            action: DataProcAction::Ignored,
            cpsr,
        })
    }

    fn eor(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(&instr.operands[2])?;

        let res_val = bd.build_xor(rn_val, shifter_op, "eor")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn mov(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let (shifter, c) = self.shifter_operand(&instr.operands[1])?;

        if instr.updates_flags {
            let n = bd.build_int_compare(IntPredicate::SLT, shifter, imm!(self, 0), "n")?;
            let z = bd.build_int_compare(IntPredicate::EQ, shifter, imm!(self, 0), "z")?;
            let cpsr = self.set_flags(Some(n), Some(z), c, None)?;
            Ok(DataProcResult {
                action: DataProcAction::single(rd, shifter),
                cpsr: Some(cpsr),
            })
        } else {
            Ok(DataProcResult {
                action: DataProcAction::single(rd, shifter),
                cpsr: None,
            })
        }
    }

    fn mvn(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let (shifter, c) = self.shifter_operand(&instr.operands[1])?;

        let res_val = bd.build_not(shifter, "mvn")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn orr(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c) = self.shifter_operand(&instr.operands[2])?;

        let res_val = bd.build_or(rn_val, shifter_op, "orr")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn rsb(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        if !instr.updates_flags {
            let res_val = bd.build_int_sub(shifter_op, rn_val, "rsb")?;
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, res_val),
                cpsr: None,
            });
        }

        let ures =
            call_intrinsic!(bd, self.usub_with_overflow, shifter_op, rn_val).into_struct_value();
        let sres =
            call_intrinsic!(bd, self.ssub_with_overflow, shifter_op, rn_val).into_struct_value();

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
            action: DataProcAction::single(rd, res_val),
            cpsr: Some(cpsr),
        })
    }

    fn rsc(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;
        let not_c = bd.build_not(c_in, "not_c")?;

        let ssub_res_1 =
            call_intrinsic!(bd, self.ssub_with_overflow, shifter_op, rn_val).into_struct_value();
        let s1 = bd
            .build_extract_value(ssub_res_1, 0, "sres1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(ssub_res_1, 1, "v1")?
            .into_int_value();

        let ssub_res_2 =
            call_intrinsic!(bd, self.ssub_with_overflow, s1, not_c).into_struct_value();
        let s2 = bd
            .build_extract_value(ssub_res_2, 0, "sres2")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(ssub_res_2, 1, "v2")?
            .into_int_value();

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, s2),
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
            action: DataProcAction::single(rd, s2),
            cpsr: Some(cpsr),
        })
    }

    fn sbc(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;
        let not_c = bd.build_not(c_in, "not_c")?;

        let ssub_res_1 =
            call_intrinsic!(bd, self.ssub_with_overflow, rn_val, shifter_op).into_struct_value();
        let s1 = bd
            .build_extract_value(ssub_res_1, 0, "sres1")?
            .into_int_value();
        let v1 = bd
            .build_extract_value(ssub_res_1, 1, "v1")?
            .into_int_value();

        let ssub_res_2 =
            call_intrinsic!(bd, self.ssub_with_overflow, s1, not_c).into_struct_value();
        let s2 = bd
            .build_extract_value(ssub_res_2, 0, "sres2")?
            .into_int_value();
        let v2 = bd
            .build_extract_value(ssub_res_2, 1, "v2")?
            .into_int_value();

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, s2),
                cpsr: None,
            });
        }

        let usub_res_1 =
            call_intrinsic!(bd, self.usub_with_overflow, rn_val, shifter_op).into_struct_value();
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
            action: DataProcAction::single(rd, s2),
            cpsr: Some(cpsr),
        })
    }

    fn sub(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(&instr.operands[2])?;

        if !instr.updates_flags {
            let sub_res = bd.build_int_sub(rn_val, shifter_op, "sub")?;
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, sub_res),
                cpsr: None,
            });
        }
        let ures =
            call_intrinsic!(bd, self.usub_with_overflow, rn_val, shifter_op).into_struct_value();
        let sres =
            call_intrinsic!(bd, self.ssub_with_overflow, rn_val, shifter_op).into_struct_value();

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
            action: DataProcAction::single(rd, sres_val),
            cpsr: Some(cpsr),
        })
    }

    fn teq(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for d because eor expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        let DataProcResult { action: _, cpsr } = self.eor(&instr)?;
        Ok(DataProcResult {
            action: DataProcAction::Ignored,
            cpsr,
        })
    }

    fn tst(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd because and expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        let DataProcResult { action: _, cpsr } = self.and(&instr)?;
        Ok(DataProcResult {
            action: DataProcAction::Ignored,
            cpsr,
        })
    }

    fn mla(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rs = instr.get_reg_op(2);
        let rn = instr.get_reg_op(3);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rn_val = self.reg_map.get(rn);
        let mul = bd.build_int_mul(rm_val, rs_val, "mul")?;
        let mla = bd.build_int_add(mul, rn_val, "mla")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, mla),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, mla),
            cpsr: Some(cpsr),
        })
    }

    fn mul(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rs = instr.get_reg_op(2);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let mul = bd.build_int_mul(rm_val, rs_val, "mul")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::single(rd, mul),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::single(rd, mul),
            cpsr: Some(cpsr),
        })
    }

    fn smlal(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        let i64_t = self.llvm_ctx.i64_type();
        let rdhi_i64 = bd.build_int_s_extend(rdhi_val, i64_t, "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, i64_t, "rdlo_i64")?;
        let acc = bd.build_left_shift(rdhi_i64, imm64!(self, 32), "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        let rm_i64 = bd.build_int_s_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        let mla_res = bd.build_int_add(mul_res, acc, "smlal")?;

        let lo_i32 = bd.build_int_truncate(mla_res, self.i32_t, "lo_i32")?;
        let hi = bd.build_right_shift(mla_res, imm64!(self, 32), true, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::double(rdhi, hi_i32, rdlo, lo_i32),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::double(rdhi, hi_i32, rdlo, lo_i32),
            cpsr: Some(cpsr),
        })
    }

    fn smull(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Sign-extend to i64 and multiply
        let rm_i64 = bd.build_int_s_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "smull")?;

        let lo_i32 = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;
        let hi = bd.build_right_shift(mul_res, imm64!(self, 32), true, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::double(rdlo, lo_i32, rdhi, hi_i32),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::double(rdlo, lo_i32, rdhi, hi_i32),
            cpsr: Some(cpsr),
        })
    }

    fn umlal(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        let i64_t = self.llvm_ctx.i64_type();
        let rdhi_i64 = bd.build_int_z_extend(rdhi_val, i64_t, "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, i64_t, "rdlo_i64")?;
        let acc = bd.build_left_shift(rdhi_i64, imm64!(self, 32), "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        let rm_i64 = bd.build_int_z_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        let mla_res = bd.build_int_add(mul_res, acc, "umlal")?;

        let lo_i32 = bd.build_int_truncate(mla_res, self.i32_t, "lo_i32")?;
        let hi = bd.build_right_shift(mla_res, imm64!(self, 32), false, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::double(rdhi, hi_i32, rdlo, lo_i32),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::double(rdhi, hi_i32, rdlo, lo_i32),
            cpsr: Some(cpsr),
        })
    }

    fn umull(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let bd = self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Sign-extend to i64 and multiply
        let rm_i64 = bd.build_int_z_extend(rm_val, self.llvm_ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.llvm_ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "smull")?;

        let lo_i32 = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;
        let hi = bd.build_right_shift(mul_res, imm64!(self, 32), false, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            return Ok(DataProcResult {
                action: DataProcAction::double(rdlo, lo_i32, rdhi, hi_i32),
                cpsr: None,
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        Ok(DataProcResult {
            action: DataProcAction::double(rdlo, lo_i32, rdhi, hi_i32),
            cpsr: Some(cpsr),
        })
    }

    fn mrs(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        let rd = instr.get_reg_op(0);
        let cpsr = self.reg_map.get(Reg::CPSR);
        Ok(DataProcResult {
            action: DataProcAction::single(rd, cpsr),
            cpsr: None,
        })
    }

    fn msr(&mut self, instr: &ArmInstruction) -> Result<DataProcResult<'a>> {
        if instr.operands[0].op_type != ArmOperandType::SysReg(RegId(9)) {
            // SysReg(RegId(9)) seems to correspond to cpsr_fc/spsr_fc, which may be the only way
            // you can call this since in ARMv4 without user mode? I still don't quite understand
            // how cpsr instructions get decoded by capstone.
            println!("{:#?}", instr);
            panic!("Unexpected operand");
        }
        let bd = self.builder;
        let update_val = match instr.operands[1].op_type {
            ArmOperandType::Reg(reg_id) => self.reg_map.get(Reg::from(reg_id)),
            ArmOperandType::Imm(imm) => imm!(self, imm),
            _ => panic!("unhandled operand type"),
        };
        let cpsr = self.reg_map.get(Reg::CPSR);
        let masked = bd.build_and(update_val, imm!(self, 0xf00000ff), "msk")?;
        let keep = bd.build_and(cpsr, imm!(self, 0x0fffff00), "keep")?;
        Ok(DataProcResult {
            action: DataProcAction::single(Reg::CPSR, bd.build_or(masked, keep, "or")?),
            cpsr: None,
        })
    }
}

#[cfg(test)]
mod tests {

    use capstone::RegId;
    use capstone::arch::arm::{ArmReg, ArmShift};
    use inkwell::context::Context;

    use super::*;
    use crate::arm::state::ArmState;
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
        let mut tst = ShifterOperandTestCase::new(&ctx, ArmShift::Rrx(1));

        tst.state.regs[Reg::CPSR] |= C.0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b11111000110011100111000000100110, true);
        assert_eq!(res, expect_res);

        tst.state.regs[Reg::CPSR] = 0;
        let res = tst.run(0b11110001100111001110000001001101, None);
        let expect_res = (0b01111000110011100111000000100110, true);
        assert_eq!(res, expect_res);
    }
}
