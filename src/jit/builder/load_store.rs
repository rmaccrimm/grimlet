#![allow(dead_code)]
use anyhow::{Result, anyhow, bail};
use capstone::arch::arm::ArmShift;
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::arm::disasm::instruction::{ArmInstruction, MemOffset, MemOperand, WritebackMode};
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

    fn exec_load_conditional<F>(&mut self, _instr: &ArmInstruction, _inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<LoadSingle<'a>>,
    {
        todo!()
    }

    fn exec_load_multiple_conditional<F>(
        &mut self,
        _instr: &ArmInstruction,
        _inner: F,
    ) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<LoadMultiple<'a>>,
    {
        todo!()
    }

    fn exec_store_conditional<F>(&mut self, _instr: &ArmInstruction, _inner: F) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<StoreSingle<'a>>,
    {
        // Stores are somewhat simple since we only have 1 register to potentially update
        todo!()
    }

    fn exec_store_multiple_conditional<F>(
        &mut self,
        _instr: &ArmInstruction,
        _inner: F,
    ) -> Result<()>
    where
        F: Fn(&mut Self, ArmInstruction) -> Result<StoreMultiple<'a>>,
    {
        // Stores are somewhat simple since we only have 1 register to potentially update
        todo!()
    }

    fn addressing_mode(&mut self, mem_op: &MemOperand) -> Result<AddrMode<'a>> {
        let bd = self.builder;
        let base_val = self.reg_map.get(mem_op.base);

        let calc_addr = match mem_op.offset {
            MemOffset::Reg {
                index,
                shift,
                subtract,
            } => {
                let index_val = self.reg_map.get(index);
                let shifted = self.imm_shift(index_val, shift)?;
                if subtract {
                    bd.build_int_sub(base_val, shifted, "addr")?
                } else {
                    bd.build_int_add(base_val, shifted, "addr")?
                }
            }
            MemOffset::Imm(i) => bd.build_int_add(base_val, imm!(self, i), "addr")?,
        };

        let addr_mode = match mem_op.writeback {
            None => AddrMode {
                writeback: None,
                addr: calc_addr,
            },
            Some(WritebackMode::PreIndex) => AddrMode {
                writeback: Some(RegUpdate {
                    reg: mem_op.base,
                    value: calc_addr,
                }),
                addr: calc_addr,
            },
            Some(WritebackMode::PostIndex) => {
                // Write back calc address, but use base as load/store
                AddrMode {
                    writeback: Some(RegUpdate {
                        reg: mem_op.base,
                        value: calc_addr,
                    }),
                    addr: base_val,
                }
            }
        };
        Ok(addr_mode)
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
            ArmShift::Rrx(_) => {
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
        fn new(context: &'ctx Context, mem_op: &MemOperand) -> Self {
            let mut compiler = Compiler::new(context);
            let mut f = compiler.new_function(0, None);

            f.load_initial_reg_values(&vec![Reg::R7, Reg::R8, Reg::R9].into_iter().collect())
                .unwrap();

            let addr_mode = f.addressing_mode(mem_op).unwrap();
            if let Some(wb) = addr_mode.writeback {
                f.reg_map.update(wb.reg, wb.value);
            }
            f.reg_map.update(Reg::R9, addr_mode.addr);

            f.write_state_out().unwrap();
            f.builder.build_return(None).unwrap();
            let f = f.compile().unwrap();
            Self {
                f,
                state: ArmState::default(),
            }
        }

        /// Set base and index (may be ignored). Return base, and calculated address
        fn run(&mut self, base: u32, index: u32) -> (u32, u32) {
            self.state.regs[Reg::R7] = base;
            self.state.regs[Reg::R8] = index;
            self.state.regs[Reg::R9] = 0;
            unsafe {
                self.f.call(&mut self.state);
            }
            // should never change
            assert_eq!(self.state.regs[Reg::R8], index);
            (self.state.regs[Reg::R7], self.state.regs[Reg::R9])
        }
    }

    fn reg_offset(subtract: bool, shift: Option<ArmShift>) -> MemOffset {
        MemOffset::Reg {
            index: Reg::R8,
            shift: match shift {
                Some(s) => s,
                None => ArmShift::Invalid,
            },
            subtract,
        }
    }

    #[test]
    fn test_addressing_mode_imm_no_writeback() {
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: MemOffset::Imm(100),
            writeback: None,
        };
        let ctx = Context::create();

        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 300));

        mem_op.offset = MemOffset::Imm(0);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 200));

        mem_op.offset = MemOffset::Imm(-20);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 180));
    }

    #[test]
    fn test_addressing_mode_reg_no_writeback() {
        let ctx = Context::create();
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: reg_offset(false, None),
            writeback: None,
        };
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 299));
        assert_eq!(tst.run(200, 0), (200, 200));
        assert_eq!(tst.run(200, 4095), (200, 4295));

        mem_op.offset = reg_offset(true, None);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 101));
        assert_eq!(tst.run(200, 0), (200, 200));
        // undeflows to -1
        assert_eq!(tst.run(200, 201), (200, 0xffffffff));
    }

    #[test]
    fn test_addressing_mode_shift_reg_no_writeback() {
        let ctx = Context::create();
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: reg_offset(false, Some(ArmShift::Lsr(3))),
            writeback: None,
        };
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 240), (200, 230)); //  240/8 = 30
        assert_eq!(tst.run(200, 0b111), (200, 200));

        mem_op.offset = reg_offset(true, Some(ArmShift::Lsl(4)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 2), (200, 168));

        mem_op.offset = reg_offset(false, Some(ArmShift::Asr(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 0x80000000), (200, 0xe00000c8));
    }

    #[test]
    fn test_addressing_mode_writeback_preindex() {
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: MemOffset::Imm(100),
            writeback: Some(WritebackMode::PreIndex),
        };
        let ctx = Context::create();

        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (300, 300));

        mem_op.offset = MemOffset::Imm(0);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 200));

        mem_op.offset = MemOffset::Imm(-20);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (180, 180));

        mem_op.offset = reg_offset(false, None);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (299, 299));

        mem_op.offset = reg_offset(true, None);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 199), (1, 1));

        mem_op.offset = reg_offset(false, Some(ArmShift::Rrx(0)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 0xf), (0x800000cf, 0x800000cf));

        mem_op.offset = reg_offset(true, Some(ArmShift::Lsr(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 0x324), (0xffffffff, 0xffffffff));
    }

    #[test]
    fn test_addressing_mode_writeback_postindex() {
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: MemOffset::Imm(55),
            writeback: Some(WritebackMode::PostIndex),
        };
        let ctx = Context::create();

        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (255, 200));

        mem_op.offset = MemOffset::Imm(-100);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (100, 200));

        mem_op.offset = MemOffset::Imm(0);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 99), (200, 200));

        mem_op.offset = reg_offset(false, None);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(0, 210), (210, 0));
        assert_eq!(tst.run(100, 150), (250, 100));

        mem_op.offset = reg_offset(true, None);
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 199), (1, 200));

        mem_op.offset = reg_offset(false, Some(ArmShift::Ror(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(0, 0x3), (0xc0000000, 0));
    }
}
