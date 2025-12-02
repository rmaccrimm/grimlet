use crate::{
    arm::{cpu::Reg, disasm::ArmDisasm},
    jit::{FunctionBuilder, builder::InstrHist},
};
use anyhow::{Result, anyhow};
use capstone::arch::arm::ArmOperandType;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, instr: &ArmDisasm) {
        // Essential a NOP, until the flags are actually used
        let reg = self.reg_map.get(instr.get_reg_op(0));
        let imm = self.i32_t.const_int(instr.get_imm_op(1) as u64, false);
        self.last_instr = InstrHist {
            opcode: instr.opcode,
            inputs: vec![reg, imm],
        };
    }

    /// TODO flags dependent on shift (is this encoded in cs instruction somehow?)
    /// TODO branch when mov'ing to PC
    pub(super) fn arm_mov(&mut self, instr: &ArmDisasm) -> Result<()> {
        let dest = instr.get_reg_op(0);
        match instr
            .operands
            .get(1)
            .ok_or(anyhow!("Missing 2nd operand"))?
            .op_type
        {
            ArmOperandType::Reg(reg_id) => {
                let src_reg = Reg::from(reg_id.0 as usize);
                let src_val = self.reg_map.get(src_reg);
                self.reg_map.update(dest, src_val);
            }
            ArmOperandType::Imm(imm) => {
                let imm_val = self.i32_t.const_int(imm as u64, false);
                self.reg_map.update(dest, imm_val);
            }
            _ => panic!("unhandled op_type"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mov() {}
}
