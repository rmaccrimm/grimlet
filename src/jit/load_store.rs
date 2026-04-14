use anyhow::{Context as _, Result, anyhow};
use inkwell::values::{BasicValue, IntValue};

use crate::arm::disasm::instruction::{ArmInstruction, MemOffset, MemOperand, WritebackMode};
use crate::arm::state::Reg;
use crate::arm::state::memory::{MemReadable, MemWriteable, MemoryManager, ReadVal, WriteVal};
use crate::jit::{FunctionBuilder, InstrEffect, InstrResult, RegUpdate};

#[derive(Copy, Clone)]
/// Result of addressing mode calculation for single loads/stores
struct AddrMode<'a> {
    writeback: Option<RegUpdate<'a>>,
    addr: IntValue<'a>,
}

impl<'a> FunctionBuilder<'_, 'a> {
    pub(super) fn arm_ldr(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u32>)
    }

    pub(super) fn arm_ldrb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u8>)
    }

    pub(super) fn arm_ldrh(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<u16>)
    }

    pub(super) fn arm_ldrsb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<i8>)
    }

    pub(super) fn arm_ldrsh(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldr::<i16>)
    }

    pub(super) fn arm_ldmia(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldmia)
    }

    pub(super) fn arm_ldmib(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldmib)
    }

    pub(super) fn arm_ldmda(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldmda)
    }

    pub(super) fn arm_ldmdb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::ldmdb)
    }

    pub(super) fn arm_str(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::str::<u32>)
    }

    pub(super) fn arm_strb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::str::<u8>)
    }

    pub(super) fn arm_strh(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::str::<u16>)
    }

    pub(super) fn arm_stmia(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::stmia)
    }

    pub(super) fn arm_stmib(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::stmib)
    }

    pub(super) fn arm_stmda(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::stmda)
    }

    pub(super) fn arm_stmdb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::stmdb)
    }

    pub(super) fn arm_push(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::push)
    }

    pub(super) fn arm_pop(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::pop)
    }

    /// Not a real `ARMv4` instruction, but Capstone decodes some instrutions like
    /// `ldr r0, =label` to this
    pub(super) fn arm_adr(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::adr)
    }

    pub(super) fn arm_swp(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::swp::<u32>)
    }

    pub(super) fn arm_swpb(&mut self, instr: &ArmInstruction) -> bool {
        exec_instr!(self, exec_conditional, instr, Self::swp::<u8>)
    }

    fn addressing_mode(&self, mem_op: &MemOperand) -> Result<AddrMode<'a>> {
        let bd = &self.builder;
        let base_val = self.reg_map.get(mem_op.base);

        let calc_addr = match mem_op.offset {
            MemOffset::Reg {
                index,
                shift,
                subtract,
            } => {
                let index_val = self.reg_map.get(index);
                let (shifted, _) = self.shift_value(index_val, shift)?;
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
                writeback: Some(RegUpdate(mem_op.base, calc_addr)),
                addr: calc_addr,
            },
            Some(WritebackMode::PostIndex) => {
                // Write back calc address, but use base as load/store
                AddrMode {
                    writeback: Some(RegUpdate(mem_op.base, calc_addr)),
                    addr: base_val,
                }
            }
        };
        Ok(addr_mode)
    }

    // read value (i32), wait states (i32)
    fn call_mem_read<T>(&self, addr: IntValue<'a>) -> Result<(IntValue<'a>, IntValue<'a>)>
    where
        T: MemReadable,
    {
        let bd = &self.builder;
        // read value, wait states
        let return_t = self
            .ctx
            .struct_type(&[self.i32_t.into(), self.i32_t.into()], false);

        let read_fn_t = return_t.fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let read_fn_ptr = self.get_external_func_pointer(
            MemoryManager::read::<T> as fn(&MemoryManager, u32) -> ReadVal as usize,
        )?;

        let result = call_indirect!(bd, read_fn_t, read_fn_ptr, self.mem_ptr, addr)
            .try_as_basic_value()
            .left()
            .ok_or_else(|| anyhow!("failed to get read return val"))?
            .into_struct_value();

        let read_val = bd
            .build_extract_value(result, 0, "rval")?
            .as_basic_value_enum()
            .into_int_value();
        let wait_states = bd
            .build_extract_value(result, 1, "wait")?
            .as_basic_value_enum()
            .into_int_value();

        Ok((read_val, wait_states))
    }

    /// Returns `WriteVal` as tuple, i.e. (`should_exit`, `wait_states`)
    fn call_mem_write<T>(
        &self,
        addr: IntValue<'a>,
        value: IntValue<'a>,
    ) -> Result<(IntValue<'a>, IntValue<'a>)>
    where
        T: MemWriteable,
    {
        let bd = &self.builder;
        let write_fn_t = self
            .ctx
            .struct_type(&[self.i8_t.into(), self.i32_t.into()], false)
            .fn_type(
                &[self.ptr_t.into(), self.i32_t.into(), self.i32_t.into()],
                false,
            );
        let write_fn_ptr = self.get_external_func_pointer(
            MemoryManager::write as fn(&mut MemoryManager, u32, T) -> WriteVal as usize,
        )?;

        let result = call_indirect!(
            self.builder,
            write_fn_t,
            write_fn_ptr,
            self.mem_ptr,
            addr,
            value
        )
        .try_as_basic_value()
        .left()
        .ok_or_else(|| anyhow!("failed to get write return val"))?
        .into_struct_value();

        let should_exit = bd
            .build_extract_value(result, 0, "exit")?
            .as_basic_value_enum()
            .into_int_value();
        let wait_states = bd
            .build_extract_value(result, 1, "wait")?
            .as_basic_value_enum()
            .into_int_value();

        Ok((should_exit, wait_states))
    }

    fn ldr<T>(&self, instr: &ArmInstruction) -> InstrResult<'a>
    where
        T: MemReadable,
    {
        let rd = instr.get_reg_op(0);
        let addr_mode: AddrMode = self.addressing_mode(&instr.get_mem_op(1)?)?;
        let (val, cycles) = self.call_mem_read::<T>(addr_mode.addr)?;

        let mut updates = vec![RegUpdate(rd, val)];
        if let Some(wb) = addr_mode.writeback {
            updates.push(wb);
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn ldmia(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = base_addr;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            // TODO - are these just additive?
            let (value, read_cycles) = self.call_mem_read::<u32>(addr)?;
            cycles = bd.build_int_add(cycles, read_cycles, "cycle")?;
            updates.push(RegUpdate(reg, value));
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(rn, addr));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn ldmib(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = base_addr;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            addr = bd.build_int_add(addr, imm!(self, 4), "ib")?;
            // TODO - are these just additive?
            let (value, read_cycles) = self.call_mem_read::<u32>(addr)?;
            cycles = bd.build_int_add(cycles, read_cycles, "cycle")?;
            updates.push(RegUpdate(reg, value));
        }
        if instr.writeback {
            updates.push(RegUpdate(rn, addr));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn ldmda(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr =
            bd.build_int_sub(base_addr, imm!(self, 4 * (reg_list.len() - 1)), "start")?;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            // TODO - are these just additive?
            let (value, read_cycles) = self.call_mem_read::<u32>(addr)?;
            cycles = bd.build_int_add(cycles, read_cycles, "cycle")?;
            updates.push(RegUpdate(reg, value));
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(
                rn,
                bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            ));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn ldmdb(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            // TODO - are these just additive?
            let (value, read_cycles) = self.call_mem_read::<u32>(addr)?;
            cycles = bd.build_int_add(cycles, read_cycles, "cycle")?;
            updates.push(RegUpdate(reg, value));
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(
                rn,
                bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            ));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn str<T>(&self, instr: &ArmInstruction) -> InstrResult<'a>
    where
        T: MemWriteable,
    {
        let rd = instr.get_reg_op(0);
        let rd_val = self.reg_map.get(rd);
        let addr_mode: AddrMode = self.addressing_mode(&instr.get_mem_op(1)?)?;
        let (exit, cycles) = self.call_mem_write::<T>(addr_mode.addr, rd_val)?;

        let mut updates = vec![];
        if let Some(wb) = addr_mode.writeback {
            updates.push(wb);
        }
        Ok(InstrEffect::with_exit(updates, cycles, exit))
    }

    fn stmia(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = base_addr;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            let value = self.reg_map.get(reg);
            let (_, write_cycles) = self.call_mem_write::<u32>(addr, value)?;
            cycles = bd.build_int_add(cycles, write_cycles, "cycles")?;
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(rn, addr));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn stmib(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = base_addr;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            addr = bd.build_int_add(addr, imm!(self, 4), "ib")?;
            let value = self.reg_map.get(reg);
            let (_, write_cycles) = self.call_mem_write::<u32>(addr, value)?;
            cycles = bd.build_int_add(cycles, write_cycles, "cycles")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(rn, addr));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn stmda(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr =
            bd.build_int_sub(base_addr, imm!(self, 4 * (reg_list.len() - 1)), "start")?;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            let value = self.reg_map.get(reg);
            let (_, write_cycles) = self.call_mem_write::<u32>(addr, value)?;
            cycles = bd.build_int_add(cycles, write_cycles, "cycles")?;
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(
                rn,
                bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            ));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    fn stmdb(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rn = instr.get_reg_op(0);
        let base_addr = self.reg_map.get(rn);
        let reg_list = instr.get_reg_list(1)?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            let value = self.reg_map.get(reg);
            let (_, write_cycles) = self.call_mem_write::<u32>(addr, value)?;
            cycles = bd.build_int_add(cycles, write_cycles, "cycles")?;
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        if instr.writeback {
            updates.push(RegUpdate(
                rn,
                bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
            ));
        }
        Ok(InstrEffect::new(updates, cycles))
    }

    // Identical to stmdb but first SP operand and writeback flag are excluded by disassembler
    fn push(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let base_addr = self.reg_map.get(Reg::SP);
        let reg_list = instr.get_reg_list(0)?;

        let mut updates = vec![];
        let mut addr = bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "start")?;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            let value = self.reg_map.get(reg);
            let (_, write_cycles) = self.call_mem_write::<u32>(addr, value)?;
            cycles = bd.build_int_add(cycles, write_cycles, "cycles")?;
            addr = bd.build_int_add(addr, imm!(self, 4), "da")?;
        }
        updates.push(RegUpdate(
            Reg::SP,
            bd.build_int_sub(base_addr, imm!(self, 4 * reg_list.len()), "wb")?,
        ));
        Ok(InstrEffect::new(updates, cycles))
    }

    // Identical to ldmia but first SP operand and writeback flag are excluded by disassembler
    fn pop(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let base_addr = self.reg_map.get(Reg::SP);
        let reg_list = instr.get_reg_list(0)?;

        let mut updates = vec![];
        let mut addr = base_addr;
        let mut cycles = imm!(self, 0);
        for &reg in &reg_list {
            let (value, read_cycles) = self.call_mem_read::<u32>(addr)?;
            cycles = bd.build_int_add(cycles, read_cycles, "cycle")?;
            updates.push(RegUpdate(reg, value));
            addr = bd.build_int_add(addr, imm!(self, 4), "addr")?;
        }
        updates.push(RegUpdate(Reg::SP, addr));
        Ok(InstrEffect::new(updates, cycles))
    }

    fn adr(&self, instr: &ArmInstruction) -> InstrResult<'a> {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let pc_offset = instr.get_imm_op(1);
        let addr = bd.build_int_add(self.reg_map.get(Reg::PC), imm!(self, pc_offset), "adr")?;
        let updates = vec![RegUpdate(rd, addr)];
        Ok(InstrEffect::new(updates, imm!(self, 1)))
    }

    fn swp<T>(&self, instr: &ArmInstruction) -> InstrResult<'a>
    where
        T: MemReadable + MemWriteable,
    {
        let bd = &self.builder;
        let rd = instr.get_reg_op(0);
        let rm = instr.get_reg_op(1);
        let rm_val = self.reg_map.get(rm);
        let mem_op = instr.get_mem_op(2)?;
        let addr = self.reg_map.get(mem_op.base);

        // TODO unaligned rotation (does this happen for ldr/str already?)
        let (load_val, read_cycles) = self.call_mem_read::<T>(addr)?;
        let (_, write_cycles) = self.call_mem_write::<T>(addr, rm_val)?;

        let updates = vec![RegUpdate(rd, load_val)];
        Ok(InstrEffect::new(
            updates,
            bd.build_int_add(read_cycles, write_cycles, "cycle")?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use inkwell::context::Context;

    use super::*;
    use crate::arm::disasm::instruction::ArmShift;
    use crate::arm::state::{ArmState, Reg};
    use crate::jit::CompiledFunction;
    use crate::jit::flags::C;

    /// Apply the given shift to R0
    fn shift_test_case(
        context: &Context,
        init: u32,
        shift: Option<ArmShift>,
        c_flag: Option<bool>,
    ) -> u32 {
        let mut f = FunctionBuilder::new(context, 0).unwrap();

        let init_regs = &vec![Reg::R0, Reg::CPSR].into_iter().collect();

        f.load_initial_reg_values(init_regs).unwrap();

        let (shifted, _) = f.shift_value(f.reg_map.get(Reg::R0), shift).unwrap();
        f.reg_map.update(Reg::R0, shifted);
        f.write_state_out(&f.reg_map).unwrap();
        f.builder.build_return(None).unwrap();
        let f = f.compile().unwrap();
        let mut state = ArmState::default();
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
            ((0xf0f0_0000, 2), 0xfc3c_0000),
            ((0x00f0_0000, 2), 0x003c_0000),
            ((0xf0f0_0000, 31), 0xffff_ffff),
            ((0xf0f0_0000, 32), 0xffff_ffff),
            ((0xf0f0_0000, 31), 0xffff_ffff),
            ((0x70f0_0000, 32), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(
                shift_test_case(&ctx, r0, Some(ArmShift::AsrImm(sh)), None),
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_asr_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, Some(ArmShift::AsrImm(0)), None);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_asr_33_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, Some(ArmShift::AsrImm(33)), None);
    }

    #[test]
    fn test_imm_shift_lsr() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f0_00e7, 1), 0x7878_0073),
            ((0xf0f0_0000, 31), 0x0000_0001),
            ((0xf0f0_0000, 32), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(
                shift_test_case(&ctx, r0, Some(ArmShift::LsrImm(sh)), None),
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_lsr_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, Some(ArmShift::LsrImm(0)), None);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_lsr_33_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0xf0f0, Some(ArmShift::LsrImm(33)), None);
    }

    #[test]
    fn test_imm_shift_lsl() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f0_00e7, 0), 0xf0f_000e7),
            ((0xf0f0_00e7, 1), 0xe1e0_01ce),
            ((0xf0f0_0001, 31), 0x8000_0000),
            ((0xf0f0_0000, 31), 0),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(
                shift_test_case(&ctx, r0, Some(ArmShift::LslImm(sh)), None),
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_lsl_32_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, Some(ArmShift::LslImm(32)), None);
    }

    #[test]
    fn test_imm_shift_ror() {
        let ctx = Context::create();
        let cases = vec![
            ((0xf0f0_00e7, 1), 0xf878_0073),
            ((0xf0f0_00e7, 31), 0xe1e0_01cf),
        ];
        for ((r0, sh), expected) in cases {
            assert_eq!(
                shift_test_case(&ctx, r0, Some(ArmShift::RorImm(sh)), None),
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_ror_0_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, Some(ArmShift::RorImm(0)), None);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_imm_shift_ror_32_panics() {
        let ctx = Context::create();
        shift_test_case(&ctx, 0, Some(ArmShift::RorImm(32)), None);
    }

    #[test]
    fn test_imm_shift_rrx() {
        let ctx = Context::create();
        assert_eq!(
            shift_test_case(&ctx, 0x80b3_0011, Some(ArmShift::Rrx), Some(true)),
            0xc059_8008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b3_0011, Some(ArmShift::Rrx), Some(true)),
            0xc059_8008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b3_0011, Some(ArmShift::Rrx), Some(false)),
            0x4059_8008
        );
        assert_eq!(
            shift_test_case(&ctx, 0x80b3_0011, Some(ArmShift::Rrx), Some(false)),
            0x4059_8008
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
            let mut f = FunctionBuilder::new(context, 0).unwrap();
            f.load_initial_reg_values(
                &vec![Reg::R7, Reg::R8, Reg::R9, Reg::CPSR]
                    .into_iter()
                    .collect(),
            )
            .unwrap();

            let addr_mode = f.addressing_mode(mem_op).unwrap();
            if let Some(RegUpdate(r, v)) = addr_mode.writeback {
                f.reg_map.update(r, v);
            }
            f.reg_map.update(Reg::R9, addr_mode.addr);

            f.write_state_out(&f.reg_map).unwrap();
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
            shift,
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
        assert_eq!(tst.run(200, 201), (200, 0xffff_ffff));
    }

    #[test]
    fn test_addressing_mode_shift_reg_no_writeback() {
        let ctx = Context::create();
        let mut mem_op = MemOperand {
            base: Reg::R7,
            offset: reg_offset(false, Some(ArmShift::LsrImm(3))),
            writeback: None,
        };
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 240), (200, 230)); //  240/8 = 30
        assert_eq!(tst.run(200, 0b111), (200, 200));

        mem_op.offset = reg_offset(true, Some(ArmShift::LslImm(4)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 2), (200, 168));

        mem_op.offset = reg_offset(false, Some(ArmShift::AsrImm(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 0x8000_0000), (200, 0xe000_00c8));
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

        mem_op.offset = reg_offset(false, Some(ArmShift::Rrx));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        tst.state.regs[Reg::CPSR] = C.0;
        assert_eq!(tst.run(200, 0xf), (0x8000_00cf, 0x8000_00cf));
        tst.state.regs[Reg::CPSR] = 0;
        assert_eq!(tst.run(200, 0xf), (207, 207));

        mem_op.offset = reg_offset(true, Some(ArmShift::LsrImm(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(200, 0x324), (0xffff_ffff, 0xffff_ffff));
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

        mem_op.offset = reg_offset(false, Some(ArmShift::RorImm(2)));
        let mut tst = AddrModeTestCase::new(&ctx, &mem_op);
        assert_eq!(tst.run(0, 0x3), (0xc000_0000, 0));
    }
}
