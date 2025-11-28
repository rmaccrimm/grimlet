use crate::{
    arm::disasm::ArmDisasm,
    jit::{FunctionBuilder, builder::InstrHist},
};
use anyhow::Result;

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, instr: &ArmDisasm) -> Result<()> {
        // Essential a NOP, until the flags are actually used
        let reg = self.reg_map.get(instr.get_reg_op(0)?);
        let imm = self.i32_t.const_int(instr.get_imm_op(1)? as u64, false);
        self.last_instr = InstrHist {
            opcode: instr.opcode,
            inputs: vec![reg, imm],
        };
        Ok(())
    }
}
