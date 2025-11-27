use crate::{arm::disasm::ArmDisasm, jit::LlvmFunction};
use anyhow::Result;

impl<'ctx, 'a> LlvmFunction<'ctx, 'a> {
    pub(super) fn arm_cmp(&mut self, _instr: &ArmDisasm) -> Result<()> {
        // Essential a NOP, until the flags are actually used
        Ok(())
    }
}
