#![allow(dead_code)]
use anyhow::{Result, anyhow, bail};
use capstone::RegId;
use capstone::arch::arm::{ArmOpMem, ArmOperand, ArmOperandType, ArmShift};
use inkwell::IntPredicate;
use inkwell::values::IntValue;

use crate::arm::cpu::Reg;
use crate::arm::disasm::{ArmInstruction, MemOperand};
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

    fn addressing_mode(&mut self, instr: &ArmInstruction) -> Result<AddrMode<'a>> {
        let bd = self.builder;
        let mem_op = &instr.operands[1];
        let mem_inner = instr.get_mem_op(1);
        let r_base = Reg::from(mem_inner.base());

        match instr.operands.get(2) {
            // post-indexed operands: [Reg, ArmOpMem (index = 0), Reg|Imm]
            Some(post_op) => {
                debug_assert!(mem_inner.index().0 == 0);

                let offset = match post_op.op_type {
                    ArmOperandType::Imm(imm) => imm!(self, imm),
                    ArmOperandType::Reg(reg_id) => {
                        let r_off = Reg::from(reg_id);
                        let r_off_val = self.reg_map.get(r_off);
                        let off = self.imm_shift(r_off_val, post_op.shift)?;
                        if mem_op.subtracted {
                            bd.build_int_neg(off, "sub_off")?
                        } else {
                            off
                        }
                    }
                    _ => bail!("unexpected post-index op_type: {:?}", post_op.op_type),
                };

                let addr = self.reg_map.get(r_base);
                let writeback_val = bd.build_int_add(addr, offset, "wb")?;

                Ok(AddrMode {
                    writeback: Some(RegUpdate {
                        reg: r_base,
                        value: writeback_val,
                    }),
                    addr,
                })
            }
            // pre-index operands: [Reg, ArmOpMem(index=0 or disp=0)]
            None => {
                let offset = if mem_inner.index().0 == 0 {
                    imm!(self, mem_inner.disp())
                } else {
                    // register
                    let r_off = Reg::from(mem_inner.index());
                    let r_off_val = self.reg_map.get(r_off);
                    let off = self.imm_shift(r_off_val, mem_op.shift)?;
                    if mem_op.subtracted {
                        bd.build_int_neg(off, "sub_off")?
                    } else {
                        off
                    }
                };

                let addr = bd.build_int_add(self.reg_map.get(r_base), offset, "addr")?;
                let writeback = instr.writeback.then_some(RegUpdate {
                    reg: r_base,
                    value: addr,
                });
                Ok(AddrMode { writeback, addr })
            }
        }
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
