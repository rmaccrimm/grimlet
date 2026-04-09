use anyhow::{Context as _, anyhow};
use capstone::arch::arm::ArmOperandType;
use inkwell::IntPredicate;

use crate::arm::disasm::instruction::{ArmInstruction, ArmShift};
use crate::arm::state::Reg;
use crate::jit::flags::C;
use crate::jit::{FunctionBuilder, InstrEffect, InstrResult, RegUpdate};

impl<'a> FunctionBuilder<'_, 'a> {
    pub(super) fn arm_adc(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::adc);
    }

    pub(super) fn arm_add(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::add);
    }

    pub(super) fn arm_and(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::and);
    }

    pub(super) fn arm_asr(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::asr);
    }

    pub(super) fn arm_bic(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::bic);
    }

    pub(super) fn arm_cmn(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::cmn);
    }

    pub(super) fn arm_cmp(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::cmp);
    }

    pub(super) fn arm_eor(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::eor);
    }

    pub(super) fn arm_lsl(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::lsl);
    }

    pub(super) fn arm_lsr(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::lsr);
    }

    pub(super) fn arm_mla(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::mla);
    }

    pub(super) fn arm_mov(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::mov);
    }

    pub(super) fn arm_mvn(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::mvn);
    }

    pub(super) fn arm_orr(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::orr);
    }

    pub(super) fn arm_ror(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ror);
    }

    pub(super) fn arm_rsb(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::rsb);
    }

    pub(super) fn arm_rsc(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::rsc);
    }

    pub(super) fn arm_sbc(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::sbc);
    }

    pub(super) fn arm_smlal(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::smlal);
    }

    pub(super) fn arm_smull(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::smull);
    }

    pub(super) fn arm_sub(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::sub);
    }

    pub(super) fn arm_teq(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::teq);
    }

    pub(super) fn arm_tst(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::tst);
    }

    pub(super) fn arm_umlal(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::umlal);
    }

    pub(super) fn arm_umull(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::umull);
    }

    pub(super) fn arm_mul(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::mul);
    }

    pub(super) fn arm_mrs(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::mrs);
    }

    pub(super) fn arm_msr(&mut self, instr: &ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::msr);
    }

    fn adc(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        let bd = &self.builder;
        let c_in = bd.build_int_cast(self.get_flag(C)?, self.i32_t, "c32")?;

        if !instr.updates_flags {
            let add_1 = bd.build_int_add(rn_val, shifter_op, "add_1")?;
            let add_2 = bd.build_int_add(add_1, c_in, "add_2")?;
            let updates = vec![RegUpdate(rd, add_2)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

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

        let updates = vec![RegUpdate(rd, s2), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn add(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        if !instr.updates_flags {
            let res = bd.build_int_add(rn_val, shifter_op, "add_res")?;
            let updates = vec![RegUpdate(rd, res)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
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

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn and(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        let res_val = bd.build_and(rn_val, shifter_op, "and")?;
        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }
        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn asr(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        self.thumb_shift(instr, ArmShift::AsrImm, ArmShift::AsrReg)
    }

    fn bic(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        let not_shifter = bd.build_not(shifter_op, "not_shift")?;
        let res_val = bd.build_and(rn_val, not_shifter, "bic")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn cmn(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        // Perform an addition but keep only the cpsr update
        let InstrEffect { updates, .. } = self.add(&instr)?;
        let updates = updates.into_iter().filter(|up| up.0 == Reg::CPSR).collect();
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn cmp(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        // Perform an subtractxion but keep only the cpsr update
        let InstrEffect { updates, .. } = self.sub(&instr)?;
        let updates = updates.into_iter().filter(|up| up.0 == Reg::CPSR).collect();
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn eor(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c_flag) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        let res_val = bd.build_xor(rn_val, shifter_op, "eor")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c_flag, None)?;

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn lsl(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        self.thumb_shift(instr, ArmShift::LslImm, ArmShift::LslReg)
    }

    fn lsr(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        self.thumb_shift(instr, ArmShift::LsrImm, ArmShift::LsrReg)
    }

    fn mov(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let (shifter, c) = self.shifter_operand(instr.get_shifter_op(1)?)?;

        let updates = if instr.updates_flags {
            let n = bd.build_int_compare(IntPredicate::SLT, shifter, imm!(self, 0), "n")?;
            let z = bd.build_int_compare(IntPredicate::EQ, shifter, imm!(self, 0), "z")?;
            let cpsr = self.set_flags(Some(n), Some(z), c, None)?;
            vec![RegUpdate(rd, shifter), RegUpdate(Reg::CPSR, cpsr)]
        } else {
            vec![RegUpdate(rd, shifter)]
        };
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn mvn(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let (shifter, c) = self.shifter_operand(instr.get_shifter_op(1)?)?;

        let res_val = bd.build_not(shifter, "mvn")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn orr(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, c) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        let res_val = bd.build_or(rn_val, shifter_op, "orr")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, res_val, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, res_val, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), c, None)?;

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn ror(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        self.thumb_shift(instr, ArmShift::RorImm, ArmShift::RorReg)
    }

    fn rsb(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        if !instr.updates_flags {
            let res_val = bd.build_int_sub(shifter_op, rn_val, "rsb")?;
            let updates = vec![RegUpdate(rd, res_val)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
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

        let updates = vec![RegUpdate(rd, res_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn rsc(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

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
            let updates = vec![RegUpdate(rd, s2)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
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

        let updates = vec![RegUpdate(rd, s2), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn sbc(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

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
            let updates = vec![RegUpdate(rd, s2)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
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

        let updates = vec![RegUpdate(rd, s2), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn sub(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rn = instr.get_reg_op(1);
        let rn_val = self.reg_map.get(rn);
        let (shifter_op, _) = self.shifter_operand(instr.get_shifter_op(2)?)?;

        if !instr.updates_flags {
            let sub_res = bd.build_int_sub(rn_val, shifter_op, "sub")?;
            let updates = vec![RegUpdate(rd, sub_res)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
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

        let updates = vec![RegUpdate(rd, sres_val), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn teq(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for d because eor expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;
        let InstrEffect { updates, .. } = self.eor(&instr)?;
        let updates = updates.into_iter().filter(|up| up.0 == Reg::CPSR).collect();
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn tst(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let mut instr = instr.clone();
        let rd_op = instr.operands[0].clone();
        let rm_op = instr.operands[1].clone();
        // Insert a dummy operand for rd because and expects it.
        instr.operands = vec![rd_op.clone(), rd_op, rm_op];
        instr.updates_flags = true;

        let InstrEffect { updates, .. } = self.and(&instr)?;
        let updates = updates.into_iter().filter(|up| up.0 == Reg::CPSR).collect();
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn mla(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
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
            let updates = vec![RegUpdate(rd, mla)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![RegUpdate(rd, mla), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn mul(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rs = instr.get_reg_op(2);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let mul = bd.build_int_mul(rm_val, rs_val, "mul")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rd, mul)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul, imm!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul, imm!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![RegUpdate(rd, mul), RegUpdate(Reg::CPSR, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn smlal(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        let i64_t = self.ctx.i64_type();
        let rdhi_i64 = bd.build_int_s_extend(rdhi_val, i64_t, "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, i64_t, "rdlo_i64")?;
        let acc = bd.build_left_shift(rdhi_i64, imm64!(self, 32), "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        let rm_i64 = bd.build_int_s_extend(rm_val, self.ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        let mla_res = bd.build_int_add(mul_res, acc, "smlal")?;

        let lo_i32 = bd.build_int_truncate(mla_res, self.i32_t, "lo_i32")?;
        let hi = bd.build_right_shift(mla_res, imm64!(self, 32), true, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rdhi, hi_i32), RegUpdate(rdlo, lo_i32)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![
            RegUpdate(rdhi, hi_i32),
            RegUpdate(rdlo, lo_i32),
            RegUpdate(Reg::CPSR, cpsr),
        ];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn smull(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Sign-extend to i64 and multiply
        let rm_i64 = bd.build_int_s_extend(rm_val, self.ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_s_extend(rs_val, self.ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "smull")?;

        let lo_i32 = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;
        let hi = bd.build_right_shift(mul_res, imm64!(self, 32), true, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rdhi, hi_i32), RegUpdate(rdlo, lo_i32)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![
            RegUpdate(rdhi, hi_i32),
            RegUpdate(rdlo, lo_i32),
            RegUpdate(Reg::CPSR, cpsr),
        ];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn umlal(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);
        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);
        let rdhi_val = self.reg_map.get(rdhi);
        let rdlo_val = self.reg_map.get(rdlo);

        let i64_t = self.ctx.i64_type();
        let rdhi_i64 = bd.build_int_z_extend(rdhi_val, i64_t, "rdhi_i64")?;
        let rdlo_i64 = bd.build_int_z_extend(rdlo_val, i64_t, "rdlo_i64")?;
        let acc = bd.build_left_shift(rdhi_i64, imm64!(self, 32), "acc_hi")?;
        let acc = bd.build_or(acc, rdlo_i64, "acc")?;

        let rm_i64 = bd.build_int_z_extend(rm_val, self.ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "mul")?;

        let mla_res = bd.build_int_add(mul_res, acc, "umlal")?;

        let lo_i32 = bd.build_int_truncate(mla_res, self.i32_t, "lo_i32")?;
        let hi = bd.build_right_shift(mla_res, imm64!(self, 32), false, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rdhi, hi_i32), RegUpdate(rdlo, lo_i32)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mla_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mla_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![
            RegUpdate(rdhi, hi_i32),
            RegUpdate(rdlo, lo_i32),
            RegUpdate(Reg::CPSR, cpsr),
        ];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn umull(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rdlo = instr.get_reg_op(0);
        let rdhi = instr.get_reg_op(1);
        let rm = instr.get_reg_op(2);
        let rs = instr.get_reg_op(3);

        let rm_val = self.reg_map.get(rm);
        let rs_val = self.reg_map.get(rs);

        // Sign-extend to i64 and multiply
        let rm_i64 = bd.build_int_z_extend(rm_val, self.ctx.i64_type(), "rm_i64")?;
        let rs_i64 = bd.build_int_z_extend(rs_val, self.ctx.i64_type(), "rs_i64")?;
        let mul_res = bd.build_int_mul(rm_i64, rs_i64, "smull")?;

        let lo_i32 = bd.build_int_truncate(mul_res, self.i32_t, "lo")?;
        let hi = bd.build_right_shift(mul_res, imm64!(self, 32), false, "hi")?;
        let hi_i32 = bd.build_int_truncate(hi, self.i32_t, "hi_i32")?;

        if !instr.updates_flags {
            let updates = vec![RegUpdate(rdhi, hi_i32), RegUpdate(rdlo, lo_i32)];
            return Ok(InstrEffect {
                updates,
                cycles: imm!(self, 1),
            });
        }

        let n = bd.build_int_compare(IntPredicate::SLT, mul_res, imm64!(self, 0), "n")?;
        let z = bd.build_int_compare(IntPredicate::EQ, mul_res, imm64!(self, 0), "z")?;
        let cpsr = self.set_flags(Some(n), Some(z), None, None)?;

        let updates = vec![
            RegUpdate(rdhi, hi_i32),
            RegUpdate(rdlo, lo_i32),
            RegUpdate(Reg::CPSR, cpsr),
        ];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    #[allow(clippy::unnecessary_wraps)]
    fn mrs(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let rd = instr.get_reg_op(0);
        let cpsr = self.reg_map.get(Reg::CPSR);
        let updates = vec![RegUpdate(rd, cpsr)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }

    fn msr(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        // SysReg parsing through capstone-rs seems to be incomplete. Just manually parse the
        // rest of the instruction here.
        let r = ((instr.binary >> 22) & 1) == 1;
        let reg = if r { Reg::SPSR } else { Reg::CPSR };
        let mut mask = 0u32;
        let f = (instr.binary >> 19) & 1 == 1;
        // Assuming f and c are the only 2 useable field specifiersa
        if f {
            mask |= 0xf000_0000;
        }
        let c = (instr.binary >> 16) & 1 == 1;
        if c {
            mask |= 0xff;
        }
        let bd = &self.builder;
        let update_val = match instr.operands[1].op_type {
            ArmOperandType::Reg(reg_id) => self.reg_map.get(Reg::from(reg_id)),
            ArmOperandType::Imm(imm) => imm!(self, imm),
            _ => panic!("unhandled operand type"),
        };
        let curr_status = self.reg_map.get(reg);
        let masked = bd.build_and(update_val, imm!(self, mask), "msk")?;
        let keep = bd.build_and(curr_status, imm!(self, 0x0fff_ff00), "keep")?;
        let res_val = bd.build_or(masked, keep, "or")?;
        let updates = vec![RegUpdate(reg, res_val)];
        Ok(InstrEffect {
            updates,
            cycles: imm!(self, 1),
        })
    }
}
