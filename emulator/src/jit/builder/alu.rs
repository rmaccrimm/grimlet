use anyhow::{anyhow, Result};
use capstone::arch::arm::ArmOperandType;
use capstone::RegId;

use crate::arm::cpu::Reg;
use crate::arm::disasm::ArmDisasm;
use crate::jit::FunctionBuilder;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, instr: &ArmDisasm) {
        // Essential a NOP, until the flags are actually used
        let reg = self.reg_map.get(instr.get_reg_op(0));
        let imm = self.i32_t.const_int(instr.get_imm_op(1) as u64, false);
        self.set_last_instr(instr, vec![reg, imm]);
        self.increment_pc(instr.mode);
    }

    /// TODO flags dependent on shift (is this encoded in cs instruction somehow?)
    /// TODO branch when mov'ing to PC
    pub(super) fn arm_mov(&mut self, instr: &ArmDisasm) {
        let build = |f: &mut Self| -> Result<()> {
            let dest = instr.get_reg_op(0);
            match instr
                .operands
                .get(1)
                .ok_or(anyhow!("Missing 2nd operand"))?
                .op_type
            {
                ArmOperandType::Reg(reg_id) => {
                    let src_reg = Reg::from(reg_id.0 as usize);
                    let src_val = f.reg_map.get(src_reg);
                    f.set_last_instr(instr, vec![f.reg_map.get(dest), src_val]);
                    f.reg_map.update(dest, src_val);
                }
                ArmOperandType::Imm(imm) => {
                    let imm_val = f.i32_t.const_int(imm as u64, false);
                    f.set_last_instr(instr, vec![f.reg_map.get(dest), imm_val]);
                    f.reg_map.update(dest, imm_val);
                }
                _ => panic!("unhandled op_type"),
            }
            f.increment_pc(instr.mode);
            Ok(())
        };
        self.exec_conditional(instr, build, false)
    }

    pub(super) fn arm_sub(&mut self, instr: &ArmDisasm) {
        let build = |f: &mut Self| -> Result<()> {
            let rd = instr.get_reg_op(0);
            let rn = instr.get_reg_op(1);
            let rn_val = f.reg_map.get(rn);
            match instr
                .operands
                .get(2)
                .ok_or(anyhow!("Missing 2nd operand"))?
                .op_type
            {
                ArmOperandType::Reg(RegId(r)) => {
                    let rm = Reg::from(r as usize);
                    let rm_val = f.reg_map.get(rm);
                    if instr.updates_flags {
                        f.set_last_instr(instr, vec![rn_val, rm_val]);
                    }
                    let sub = f.builder.build_int_sub(rn_val, rm_val, "sub")?;
                    f.reg_map.update(rd, sub);
                }
                ArmOperandType::Imm(imm) => {
                    let imm_val = f.i32_t.const_int(imm as u64, false);
                    if instr.updates_flags {
                        f.set_last_instr(instr, vec![rn_val, imm_val]);
                    }
                    let sub = f.builder.build_int_sub(rn_val, imm_val, "sub")?;
                    f.reg_map.update(rd, sub);
                }
                _ => panic!("unhandled op_type"),
            }
            Ok(())
        };
        self.exec_conditional(instr, build, false);
    }

    pub(super) fn arm_mul(&mut self, instr: &ArmDisasm) {
        let build = |f: &mut Self| -> Result<()> {
            let rd = instr.get_reg_op(0);
            let rm = instr.get_reg_op(1);
            let rs = instr.get_reg_op(2);
            let rm_val = f.reg_map.get(rm);
            let rs_val = f.reg_map.get(rs);
            if instr.updates_flags {
                f.set_last_instr(instr, vec![rm_val, rs_val]);
            }
            let res = f.builder.build_int_mul(rm_val, rs_val, "mul")?;
            f.reg_map.update(rd, res);
            Ok(())
        };
        self.exec_conditional(instr, build, false);
    }
}

#[cfg(test)]
mod tests {}
