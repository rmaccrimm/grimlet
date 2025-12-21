use anyhow::{Context as _, Result, anyhow, bail};
use capstone::arch::arm::ArmShift;
use inkwell::values::IntValue;

use crate::arm::disasm::instruction::{ArmInstruction, MemOffset, MemOperand, WritebackMode};
use crate::arm::state::Reg;
use crate::arm::state::memory::{MainMemory, MemReadable, MemWriteable};
use crate::jit::builder::flags::C;
use crate::jit::builder::{FunctionBuilder, RegUpdate};

/// Result of addressing mode calculation for single loads/stores
struct AddrMode<'a> {
    writeback: Option<RegUpdate<'a>>,
    addr: IntValue<'a>,
}

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn arm_ldr(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u32>)
    }

    pub(super) fn arm_ldrb(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u8>)
    }

    pub(super) fn arm_ldrh(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u16>)
    }

    pub(super) fn arm_ldrsb(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<i8>)
    }

    pub(super) fn arm_ldrsh(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<i16>)
    }

    pub(super) fn arm_ldmia(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldmia)
    }

    pub(super) fn arm_ldmib(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldmib)
    }

    pub(super) fn arm_ldmda(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldmda)
    }

    pub(super) fn arm_ldmdb(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::ldmdb)
    }

    pub(super) fn arm_str(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::str::<u32>)
    }

    pub(super) fn arm_strb(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::str::<u8>)
    }

    pub(super) fn arm_strh(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::str::<u16>)
    }

    pub(super) fn arm_stmia(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::stmia)
    }

    pub(super) fn arm_stmib(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::stmib)
    }

    pub(super) fn arm_stmda(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::stmda)
    }

    pub(super) fn arm_stmdb(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::stmdb)
    }

    pub(super) fn arm_push(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::push)
    }

    pub(super) fn arm_pop(&mut self, instr: ArmInstruction) {
        exec_instr!(self, exec_conditional, instr, Self::pop)
    }

    fn addressing_mode(&self, mem_op: &MemOperand) -> Result<AddrMode<'a>> {
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
                    Ok(imm!(self, 0))
                } else {
                    bd.build_right_shift(value, imm!(self, imm), false, "lsr")
                }
            }
            ArmShift::Ror(imm) => {
                debug_assert!(imm > 0 && imm < 32);
                Ok(call_intrinsic!(bd, self.fshr, value, value, imm!(self, imm)).into_int_value())
            }
            ArmShift::Rrx(_) => {
                let c = bd.build_int_cast_sign_flag(self.get_flag(C)?, self.i32_t, false, "c")?;
                Ok(call_intrinsic!(bd, self.fshr, c, value, imm!(self, 1)).into_int_value())
            }
            _ => bail!("unsupported shift type for memory access: {:?}", shift),
        }?;
        Ok(shifted)
    }

    fn ldr<T>(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>>
    where
        T: MemReadable,
    {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let addr_mode: AddrMode = self.addressing_mode(&instr.get_mem_op()?)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<T> as fn(&MainMemory, u32) -> u32 as usize,
        )?;
        let load_val =
            call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr_mode.addr);

        let mut updates = vec![RegUpdate {
            reg: rd,
            value: load_val,
        }];
        if let Some(wb) = addr_mode.writeback {
            updates.push(wb);
        }
        Ok(updates)
    }

    fn ldmia(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<u32> as fn(&MainMemory, u32) -> u32 as usize,
        )?;

        let mut updates = vec![];
        let mut addr = base_addr;
        for &reg in reg_list.iter() {
            let value = call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr);
            updates.push(RegUpdate { reg, value });
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: addr,
            })
        }
        Ok(updates)
    }

    fn ldmib(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<u32> as fn(&MainMemory, u32) -> u32 as usize,
        )?;

        let mut updates = vec![];
        let mut addr = base_addr;
        for &reg in reg_list.iter() {
            addr = bd.build_int_add(addr, imm!(self, 4), "ib")?;
            let value = call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr);
            updates.push(RegUpdate { reg, value });
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: addr,
            })
        }
        Ok(updates)
    }

    fn ldmda(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<u32> as fn(&MainMemory, u32) -> u32 as usize,
        )?;

        let mut updates = vec![];
        let mut addr =
            bd.build_int_sub(base_addr, imm!(self, 4 * (reg_list.len() - 1)), "start")?;
        for &reg in reg_list.iter() {
            let value = call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr);
            updates.push(RegUpdate { reg, value });
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            })
        }
        Ok(updates)
    }

    fn ldmdb(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<u32> as fn(&MainMemory, u32) -> u32 as usize,
        )?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        for &reg in reg_list.iter() {
            let value = call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr);
            updates.push(RegUpdate { reg, value });
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            })
        }
        Ok(updates)
    }

    fn str<T>(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>>
    where
        T: MemWriteable,
    {
        let bd = self.builder;
        let rd = instr.get_reg_op(0);
        let rd_val = self.reg_map.get(rd);
        let addr_mode: AddrMode = self.addressing_mode(&instr.get_mem_op()?)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self
            .get_external_func_pointer(MainMemory::write as fn(&mut MainMemory, u32, T) as usize)?;

        call_indirect!(
            bd,
            write_fn_t,
            write_fn_ptr,
            self.mem_ptr,
            addr_mode.addr,
            rd_val
        );
        Ok(match addr_mode.writeback {
            Some(wb) => vec![wb],
            None => vec![],
        })
    }

    fn stmia(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self.get_external_func_pointer(
            MainMemory::write as fn(&mut MainMemory, u32, u32) as usize,
        )?;

        let mut updates = vec![];
        let mut addr = base_addr;
        for &reg in reg_list.iter() {
            let value = self.reg_map.get(reg);
            call_indirect!(bd, write_fn_t, write_fn_ptr, self.mem_ptr, addr, value);
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: addr,
            })
        }
        Ok(updates)
    }

    fn stmib(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self.get_external_func_pointer(
            MainMemory::write as fn(&mut MainMemory, u32, u32) as usize,
        )?;

        let mut updates = vec![];
        let mut addr = base_addr;
        for &reg in reg_list.iter() {
            addr = bd.build_int_add(addr, imm!(self, 4), "ib")?;
            let value = self.reg_map.get(reg);
            call_indirect!(bd, write_fn_t, write_fn_ptr, self.mem_ptr, addr, value);
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: addr,
            })
        }
        Ok(updates)
    }

    fn stmda(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self.get_external_func_pointer(
            MainMemory::write as fn(&mut MainMemory, u32, u32) as usize,
        )?;

        let mut updates = vec![];
        let mut addr =
            bd.build_int_sub(base_addr, imm!(self, 4 * (reg_list.len() - 1)), "start")?;
        for &reg in reg_list.iter() {
            let value = self.reg_map.get(reg);
            call_indirect!(bd, write_fn_t, write_fn_ptr, self.mem_ptr, addr, value);
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            })
        }
        Ok(updates)
    }

    fn stmdb(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self.get_external_func_pointer(
            MainMemory::write as fn(&mut MainMemory, u32, u32) as usize,
        )?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        for &reg in reg_list.iter() {
            let value = self.reg_map.get(reg);
            call_indirect!(bd, write_fn_t, write_fn_ptr, self.mem_ptr, addr, value);
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate {
                reg: rn,
                value: bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            })
        }
        Ok(updates)
    }

    // Identical to stmdb but first SP operand and writeback flag are excluded by disassembler
    fn push(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let base_addr = self.reg_map.get(Reg::SP);
        let reg_list = instr.get_reg_list(0)?;

        let write_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
            false,
        );
        let write_fn_ptr = self.get_external_func_pointer(
            MainMemory::write as fn(&mut MainMemory, u32, u32) as usize,
        )?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        for &reg in reg_list.iter() {
            let value = self.reg_map.get(reg);
            call_indirect!(bd, write_fn_t, write_fn_ptr, self.mem_ptr, addr, value);
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        updates.push(RegUpdate {
            reg: Reg::SP,
            value: bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
        });
        Ok(updates)
    }

    // Identical to ldmia but first SP operand and writeback flag are excluded by disassembler
    fn pop(&self, instr: &ArmInstruction) -> Result<Vec<RegUpdate<'a>>> {
        let bd = self.builder;
        let base_addr = self.reg_map.get(Reg::SP);
        let reg_list = instr.get_reg_list(0)?;

        let read_fn_t = self
            .i32_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MainMemory::read::<u32> as fn(&MainMemory, u32) -> u32 as usize,
        )?;

        let mut updates = vec![];
        let mut addr = base_addr;
        for &reg in reg_list.iter() {
            let value = call_indirect_with_return!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr);
            updates.push(RegUpdate { reg, value });
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        updates.push(RegUpdate {
            reg: Reg::SP,
            value: addr,
        });
        Ok(updates)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use inkwell::context::Context;

    use super::*;
    use crate::arm::state::{ArmState, Reg};
    use crate::jit::{CompiledFunction, Compiler};

    /// Apply the given shift to R0
    fn shift_test_case(context: &Context, init: u32, shift: ArmShift, c_flag: Option<bool>) -> u32 {
        let mut compiler = Compiler::new(context);
        let mut f = compiler.new_function(0, None);

        let init_regs = &vec![Reg::R0, Reg::CPSR].into_iter().collect();

        f.load_initial_reg_values(init_regs).unwrap();

        let shifted = f.imm_shift(f.reg_map.get(Reg::R0), shift).unwrap();
        f.reg_map.update(Reg::R0, shifted);
        f.write_state_out().unwrap();
        f.builder.build_return(None).unwrap();
        let f = f.compile().unwrap();
        let (tx, _) = mpsc::channel();
        let mut state = ArmState::new(tx);
        state.regs[Reg::R0] = init;
        if let Some(c) = c_flag {
            state.regs[Reg::CPSR] = if c { C.0 } else { 0 }
        }
        unsafe {
            f.call(&mut state);
        }
        state.regs[Reg::R0]
    }

    #[test]
    fn test_imm_shift_asr() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f00000, 2), 0xfc3c0000),
            ((0x00f00000, 2), 0x003c0000),
            ((0xf0f00000, 31), 0xffffffff),
            ((0xf0f00000, 32), 0xffffffff),
            ((0xf0f00000, 31), 0xffffffff),
            ((0x70f00000, 32), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(shift_test_case(&ctx, r0, ArmShift::Asr(sh), None), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_asr_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, ArmShift::Asr(0), None);
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_asr_33_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, ArmShift::Asr(33), None);
    }

    #[test]
    fn test_imm_shift_lsr() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f000e7, 1), 0x78780073),
            ((0xf0f00000, 31), 0x00000001),
            ((0xf0f00000, 32), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(shift_test_case(&ctx, r0, ArmShift::Lsr(sh), None), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_lsr_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, ArmShift::Lsr(0), None);
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_lsr_33_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, ArmShift::Lsr(33), None);
    }

    #[test]
    fn test_imm_shift_lsl() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f000e7, 0), 0xf0f000e7),
            ((0xf0f000e7, 1), 0xe1e001ce),
            ((0xf0f00001, 31), 0x80000000),
            ((0xf0f00000, 31), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(shift_test_case(&ctx, r0, ArmShift::Lsl(sh), None), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_lsl_32_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, ArmShift::Lsl(32), None);
    }

    #[test]
    fn test_imm_shift_ror() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f000e7, 1), 0xf8780073),
            ((0xf0f000e7, 31), 0xe1e001cf),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(shift_test_case(&ctx, r0, ArmShift::Ror(sh), None), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_ror_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, ArmShift::Ror(0), None);
    }

    #[test]
    #[should_panic]
    fn test_imm_shift_ror_32_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, ArmShift::Ror(32), None);
    }

    #[test]
    fn test_imm_shift_rrx() {
        let ctx = Context::create();
        assert_eq!(
            shift_test_case(&ctx, 0x80b30011, ArmShift::Rrx(0), Some(true)),
            0xc0598008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b30011, ArmShift::Rrx(1), Some(true)),
            0xc0598008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b30011, ArmShift::Rrx(0), Some(false)),
            0x40598008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b30011, ArmShift::Rrx(99), Some(false)),
            0x40598008
        );
    }

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

            f.load_initial_reg_values(
                &vec![Reg::R7, Reg::R8, Reg::R9, Reg::CPSR]
                    .into_iter()
                    .collect(),
            )
            .unwrap();

            let addr_mode = f.addressing_mode(mem_op).unwrap();
            if let Some(wb) = addr_mode.writeback {
                f.reg_map.update(wb.reg, wb.value);
            }
            f.reg_map.update(Reg::R9, addr_mode.addr);

            f.write_state_out().unwrap();
            f.builder.build_return(None).unwrap();
            let f = f.compile().unwrap();
            let (tx, _) = mpsc::channel();
            Self {
                f,
                state: ArmState::new(tx),
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
        tst.state.regs[Reg::CPSR] = C.0;
        assert_eq!(tst.run(200, 0xf), (0x800000cf, 0x800000cf));
        tst.state.regs[Reg::CPSR] = 0;
        assert_eq!(tst.run(200, 0xf), (207, 207));

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
