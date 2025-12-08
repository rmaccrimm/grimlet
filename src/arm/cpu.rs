use std::fs;
use std::ops::{Index, IndexMut};
use std::slice::Chunks;

use anyhow::{Result, anyhow};
use capstone::arch::arm::ArmReg;
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::StructType;

#[repr(C)]
pub struct MainMemory {
    pub bios: Vec<u8>,
}

impl Default for MainMemory {
    fn default() -> Self {
        // 16 kB
        let bios = vec![0; 0x4000];
        Self { bios }
    }
}

impl MainMemory {
    pub fn iter_word(&self, start_addr: usize) -> Chunks<'_, u8> {
        if !start_addr.is_multiple_of(4) {
            panic!("Mis-alligned word address: {:x}", start_addr);
        }
        match start_addr {
            0..0x3ffc => self.bios[start_addr..].chunks(4),
            _ => panic!("Address out of range: {:x}", start_addr),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmMode {
    ARM,
    THUMB,
}

impl ArmMode {
    pub fn pc_byte_offset(&self) -> u32 {
        match self {
            ArmMode::ARM => 8,
            ArmMode::THUMB => 4,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Reg {
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R5 = 5,
    R6 = 6,
    R7 = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    SP = 13,
    LR = 14,
    PC = 15,
    CPSR = 16,
}

impl From<capstone::RegId> for Reg {
    fn from(value: capstone::RegId) -> Self {
        match value.0 as u32 {
            ArmReg::ARM_REG_R0 => Reg::R0,
            ArmReg::ARM_REG_R1 => Reg::R1,
            ArmReg::ARM_REG_R2 => Reg::R2,
            ArmReg::ARM_REG_R3 => Reg::R3,
            ArmReg::ARM_REG_R4 => Reg::R4,
            ArmReg::ARM_REG_R5 => Reg::R5,
            ArmReg::ARM_REG_R6 => Reg::R6,
            ArmReg::ARM_REG_R7 => Reg::R7,
            ArmReg::ARM_REG_R8 => Reg::R8,
            ArmReg::ARM_REG_R9 => Reg::R9,
            ArmReg::ARM_REG_R10 => Reg::R10,
            ArmReg::ARM_REG_R11 => Reg::R11,
            ArmReg::ARM_REG_R12 => Reg::R12,
            ArmReg::ARM_REG_SP => Reg::SP,
            ArmReg::ARM_REG_LR => Reg::LR,
            ArmReg::ARM_REG_PC => Reg::PC,
            // Not 100% sure about this one
            ArmReg::ARM_REG_CPSR | ArmReg::ARM_REG_APSR => Reg::CPSR,
            _ => panic!("unhandled register id: {}", value.0),
        }
    }
}

impl Index<Reg> for [u32] {
    type Output = u32;

    fn index(&self, index: Reg) -> &Self::Output { &self[index as usize] }
}

impl IndexMut<Reg> for [u32] {
    fn index_mut(&mut self, index: Reg) -> &mut Self::Output { &mut self[index as usize] }
}

pub const NUM_REGS: usize = 17;

pub const REG_ITEMS: [Reg; NUM_REGS] = [
    Reg::R0,
    Reg::R1,
    Reg::R2,
    Reg::R3,
    Reg::R4,
    Reg::R5,
    Reg::R6,
    Reg::R7,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
    Reg::R12,
    Reg::SP,
    Reg::LR,
    Reg::PC,
    Reg::CPSR,
];

/// Emulated CPU state and interpreter
#[repr(C)]
pub struct ArmState {
    pub mode: ArmMode,
    pub regs: [u32; 17],
    pub mem: MainMemory,
}

impl Default for ArmState {
    fn default() -> Self {
        let mut regs = [0; NUM_REGS];
        regs[Reg::PC] = 8; // points 2 instructions ahead
        Self {
            mode: ArmMode::ARM,
            regs,
            mem: MainMemory::default(),
        }
    }
}

impl ArmState {
    // The compiler code that performs lookups into this object needs to be kept in sync with
    // the actual definition so define its corresponding LLVM type here.
    pub fn get_llvm_type<'ctx>(llvm_ctx: &'ctx Context) -> StructType<'ctx> {
        assert_eq!(size_of::<ArmMode>(), 1);
        llvm_ctx.struct_type(
            &[
                // mode
                llvm_ctx.i8_type().into(),
                // regs
                llvm_ctx.i32_type().array_type(NUM_REGS as u32).into(),
                // mem
                llvm_ctx.ptr_type(AddressSpace::default()).into(),
            ],
            false,
        )
    }

    pub fn with_bios(bios_path: &str) -> Result<Self> {
        if !fs::exists(bios_path)? {
            return Err(anyhow!("BIOS file not found"));
        }
        let bios = fs::read(bios_path)?;

        Ok(Self {
            mode: ArmMode::ARM,
            regs: [0; 17],
            mem: MainMemory { bios },
        })
    }

    pub fn curr_instr_addr(&self) -> usize {
        (self.regs[Reg::PC] - self.mode.pc_byte_offset()) as usize
    }

    pub fn r0(&self) -> u32 { self.regs[Reg::R0] }

    pub fn r1(&self) -> u32 { self.regs[Reg::R1] }

    pub fn cpsr(&self) -> u32 { self.regs[Reg::CPSR as usize] }

    pub fn jump_to(&mut self, addr: u32) {
        println!("JUMPING TO: {}", addr);
        self.regs[Reg::PC] = addr + self.mode.pc_byte_offset();
    }
}
