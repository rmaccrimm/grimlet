#![allow(dead_code)]
use anyhow::{Result, anyhow, bail};
use capstone::RegId;
use capstone::arch::arm::{ArmOpMem, ArmOperand, ArmOperandType, ArmShift};
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::arm::disasm::{ArmInstruction, MemOffset, MemOperand};
use crate::jit::builder::FunctionBuilder;
use crate::jit::builder::alu::RegUpdate;

/// Result of addressing mode calculation for single loads/stores
struct AddrMode<'a> {
    writeback: Option<RegUpdate<'a>>,
    addr: IntValue<'a>,
}

/// A register and the address to load its value from
struct RegLoad<'a> {
    reg: Reg,
    addr: IntValue<'a>,
}

/// An address and the value to store in it
struct MemStore<'a> {
    addr: IntValue<'a>,
    value: IntValue<'a>,
}

/// Describes a load instruction to be conditionally executed
struct LoadSingle<'a> {
    load: RegLoad<'a>,
    writeback: Option<RegUpdate<'a>>,
}

/// Describes a load multiple instruction to be conditionally executed
struct LoadMultiple<'a> {
    loads: Vec<RegLoad<'a>>,
    writeback: Option<RegUpdate<'a>>,
}

/// Describes a store instruction to be conditionally executed
struct StoreSingle<'a> {
    stores: MemStore<'a>,
    writeback: Option<RegUpdate<'a>>,
}

/// Describes a store multiple instruction to be conditionally executed
struct StoreMultiple<'a> {
    stores: Vec<MemStore<'a>>,
    writeback: Option<RegUpdate<'a>>,
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_ldmia(&mut self, _instr: ArmInstruction) { todo!() }

    pub(super) fn arm_stmdb(&mut self, _instr: ArmInstruction) { todo!() }

    pub(super) fn arm_stmia(&mut self, _instr: ArmInstruction) { todo!() }

    fn exec_load_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<LoadSingle<'a>>,
    {
        todo!()
    }

    fn exec_load_multiple_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<LoadMultiple<'a>>,
    {
        todo!()
    }

    fn exec_store_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<StoreSingle<'a>>,
    {
        // Stores are somewhat simple since we only have 1 register to potentially update
        todo!()
    }

    fn exec_store_multiple_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<StoreMultiple<'a>>,
    {
        // Stores are somewhat simple since we only have 1 register to potentially update
        todo!()
    }

    fn addressing_mode(
        &mut self,
        mem_op: MemOperand,
        post_index: Option<ArmOperand>,
    ) -> Result<AddrMode<'a>> {
        todo!();
    }

    fn imm_shift(&self, value: IntValue<'a>, shift: ArmShift) -> Result<IntValue<'a>> {
        let bd = self.builder;
        let shifted = match shift {
            ArmShift::Invalid => Ok(value),
            ArmShift::Asr(imm) => {
                debug_assert!(imm > 0 && imm <= 32);
                if imm == 32 {
                    bd.build_right_shift(value, imm!(self, 31), true, "asr")
                } else {
                    bd.build_right_shift(value, imm!(self, imm), true, "asr")
                }
            }
            ArmShift::Lsl(imm) => {
                debug_assert!(imm < 32);
                bd.build_left_shift(value, imm!(self, imm), "lsl")
            }
            ArmShift::Lsr(imm) => {
                debug_assert!(imm > 0 && imm <= 32);
                if imm == 32 {
                    Ok(value)
                } else {
                    bd.build_right_shift(value, imm!(self, imm), true, "asr")
                }
            }
            ArmShift::Ror(imm) => {
                debug_assert!(imm > 0 && imm < 32);
                Ok(call_intrinsic!(bd, self.fshr, value, value, imm!(self, imm)).into_int_value())
            }
            ArmShift::Rrx(imm) => {
                debug_assert_eq!(imm, 1);
                Ok(call_intrinsic!(bd, self.fshr, value, value, imm!(self, 1)).into_int_value())
            }
            _ => bail!("unsupported shift type for memory access: {:?}", shift),
        }?;
        Ok(shifted)
    }
}

#[cfg(test)]
mod tests {
    use inkwell::context::Context;

    use super::*;
    use crate::arm::cpu::ArmState;
    use crate::jit::{CompiledFunction, Compiler};

    struct AddrModeTestCase<'ctx> {
        f: CompiledFunction<'ctx>,
        state: ArmState,
    }

    impl<'ctx> AddrModeTestCase<'ctx> {
        /// r7: base register
        /// r8: index
        /// r9: stores calculated address
        fn new(context: &'ctx Context, instr: ArmInstruction) -> Self {
            let mut compiler = Compiler::new(context);
            let mut f = compiler.new_function(0, None);

            f.load_initial_reg_values(&vec![Reg::R7, Reg::R8, Reg::R9].into_iter().collect())
                .unwrap();

            // let addr_mode = f.addressing_mode(&instr).unwrap();
            // if let Some(wb) = addr_mode.writeback {
            //     f.reg_map.update(wb.reg, wb.value);
            // }
            // f.reg_map.update(Reg::R9, addr_mode.addr);

            f.write_state_out().unwrap();
            f.builder.build_return(None).unwrap();
            let f = f.compile().unwrap();
            Self {
                f,
                state: ArmState::default(),
            }
        }

        /// Set base and index (may be ignored). Return base and calculated address
        fn run(&mut self, base: u32, index: u32) -> (u32, u32) {
            self.state.regs[Reg::R7] = base;
            self.state.regs[Reg::R8] = index;
            self.state.regs[Reg::R9] = 0;
            unsafe {
                self.f.call(&mut self.state);
            }
            (self.state.regs[Reg::R8], self.state.regs[Reg::R9])
        }
    }

    #[test]
    fn test_addressing_mode_imm() {}

    #[test]
    fn test_addressing_mode_reg() {}

    #[test]
    fn test_addressing_mode_shift_reg() {}

    #[test]
    fn test_addressing_mode_pre_imm() {}

    #[test]
    fn test_addressing_mode_pre_reg() {}

    #[test]
    fn test_addressing_mode_pre_shift_reg() {}

    #[test]
    fn test_addressing_mode_post_imm() {}

    #[test]
    fn test_addressing_mode_post_reg() {}

    #[test]
    fn test_addressing_mode_post_shift_reg() {}
}
