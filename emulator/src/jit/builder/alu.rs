use anyhow::{Result, anyhow};
use capstone::RegId;
use capstone::arch::arm::{ArmCC, ArmOperandType};
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::arm::disasm::ArmInstruction;
use crate::jit::FunctionBuilder;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, instr: &ArmInstruction) {
        // Essential a NOP, until the flags are actually used
        let reg = self.reg_map.get(instr.get_reg_op(0));
        let imm = self.i32_t.const_int(instr.get_imm_op(1) as u64, false);
        self.set_last_instr(instr, vec![reg, imm]);
        self.increment_pc(instr.mode);
    }

    /// TODO flags dependent on shift (is this encoded in cs instruction somehow?)
    /// TODO branch when mov'ing to PC
    pub(super) fn arm_mov(&mut self, instr: &ArmInstruction) {
        let build = |f: &mut Self| -> Result<IntValue<'a>> {
            Ok(
                match instr
                    .operands
                    .get(1)
                    .ok_or(anyhow!("Missing 2nd operand"))?
                    .op_type
                {
                    ArmOperandType::Reg(reg_id) => {
                        let src_reg = Reg::from(reg_id.0 as usize);
                        f.reg_map.get(src_reg)
                        // f.set_last_instr(instr, vec![f.reg_map.get(dest), src_val]);
                        // f.reg_map.update(dest, src_val);
                    }
                    ArmOperandType::Imm(imm) => {
                        // f.set_last_instr(instr, vec![f.reg_map.get(dest), imm_val]);
                        // f.reg_map.update(dest, imm_val);
                        f.i32_t.const_int(imm as u64, false)
                    }
                    _ => panic!("unhandled op_type"),
                },
            )
        };
        self.exec_alu_conditional(instr, build)
    }

    pub(super) fn arm_sub(&mut self, instr: &ArmInstruction) {
        let build = |f: &mut Self| -> Result<IntValue<'a>> {
            let rn = instr.get_reg_op(1);
            let rn_val = f.reg_map.get(rn);
            let res = match instr
                .operands
                .get(2)
                .ok_or(anyhow!("Missing 2nd operand"))?
                .op_type
            {
                ArmOperandType::Reg(RegId(r)) => {
                    let rm = Reg::from(r as usize);
                    let rm_val = f.reg_map.get(rm);
                    f.builder.build_int_sub(rn_val, rm_val, "sub")
                }
                ArmOperandType::Imm(imm) => {
                    let imm_val = f.i32_t.const_int(imm as u64, false);
                    f.builder.build_int_sub(rn_val, imm_val, "sub")
                }
                _ => panic!("unhandled op_type"),
            }?;
            Ok(res)
        };
        self.exec_alu_conditional(instr, build);
    }

    pub(super) fn arm_mul(&mut self, instr: &ArmInstruction) {
        let build = |f: &mut Self| -> Result<IntValue<'a>> {
            let rm = instr.get_reg_op(1);
            let rs = instr.get_reg_op(2);
            let rm_val = f.reg_map.get(rm);
            let rs_val = f.reg_map.get(rs);
            let res = f.builder.build_int_mul(rm_val, rs_val, "mul")?;
            Ok(res)
        };
        self.exec_alu_conditional(instr, build);
    }

    /// Wraps a function for emitting an instruction in a conditional block, evaluates flags and
    /// executes based on instruction condition Leaves the builder positioned in the else block and
    /// emits code to increment program counter.
    fn exec_alu_conditional<F>(&mut self, instr: &ArmInstruction, inner: F)
    where
        F: Fn(&mut Self) -> Result<IntValue<'a>>,
    {
        if instr.cond == ArmCC::ARM_CC_AL {
            inner(self).expect("LLVM codegen failed");
            self.increment_pc(instr.mode);
            return;
        }

        let build = |f: &mut Self| -> Result<()> {
            let ctx = f.llvm_ctx;
            let bd = f.builder;
            let if_block = ctx.append_basic_block(f.func, "if");
            let end_block = ctx.append_basic_block(f.func, "end");
            // If not executing instruction, init dest value is kept;
            let rd = instr.get_reg_op(0);
            let dest_init = f.reg_map.get(rd);
            let cond = f.get_cond_value(instr.cond);
            bd.build_conditional_branch(cond, if_block, end_block)?;
            bd.position_at_end(if_block);
            let dest_calc = inner(f)?;
            bd.build_unconditional_branch(end_block)?;
            bd.position_at_end(end_block);
            let phi = bd.build_phi(f.i32_t, "phi")?;
            phi.add_incoming(&[(&dest_init, f.current_block), (&dest_calc, if_block)]);
            f.reg_map.update(rd, phi.as_basic_value().into_int_value());
            f.increment_pc(instr.mode);
            f.current_block = end_block;
            Ok(())
        };
        build(self).expect("LLVM codegen failed");
    }
}

#[cfg(test)]
mod tests {}
