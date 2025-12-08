use anyhow::{Context as _, Result};
use capstone::RegId;
use capstone::arch::arm::{ArmCC, ArmOperand, ArmOperandType};
use inkwell::IntPredicate;
use inkwell::values::{IntValue, StructValue};

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

    // Returns (i32, i1) - the value of the operand and the shifter carry-out value
    fn shifter_operand(&self, operand: &ArmOperand) -> Result<(IntValue<'a>, IntValue<'a>)> {
        let bd = self.builder;
        match operand.op_type {
            ArmOperandType::Reg(reg_id) => match operand.shift {
                capstone::arch::arm::ArmShift::Invalid => {
                    Ok((self.reg_map.get(Reg::from(reg_id)), self.get_flag(C)?))
                }
                capstone::arch::arm::ArmShift::Asr(s) => todo!(),
                capstone::arch::arm::ArmShift::Lsl(s) => todo!(),
                capstone::arch::arm::ArmShift::Lsr(s) => todo!(),
                capstone::arch::arm::ArmShift::Ror(s) => todo!(),
                capstone::arch::arm::ArmShift::Rrx(s) => todo!(),
                capstone::arch::arm::ArmShift::AsrReg(reg_id) => todo!(),
                capstone::arch::arm::ArmShift::LslReg(reg_id) => todo!(),
                capstone::arch::arm::ArmShift::LsrReg(reg_id) => todo!(),
                capstone::arch::arm::ArmShift::RorReg(reg_id) => todo!(),
                capstone::arch::arm::ArmShift::RrxReg(reg_id) => todo!(),
            },
            ArmOperandType::Imm(imm) => {
                let imm_val = self.i32_t.const_int(imm as u64, false);
                let shifted =
                    bd.build_right_shift(imm_val, self.i32_t.const_int(31, false), false, "sh")?;
                let imm_msb =
                    bd.build_int_cast_sign_flag(shifted, self.bool_t, false, "imm_msb")?;
                Ok((imm_val, imm_msb))
            }
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
            let cpsr = self.set_flags(Some(n), Some(z), Some(c), None)?;
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
mod tests {}
