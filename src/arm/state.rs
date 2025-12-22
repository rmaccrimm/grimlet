pub mod memory;

use std::ops::{Index, IndexMut};
use std::sync::mpsc::Sender;

use capstone::arch::arm::ArmReg;

use crate::arm::state::memory::MainMemory;
use crate::emulator::SystemMessage;

/// Emulated CPU state (and interpreter?)
#[repr(C)]
pub struct ArmState {
    pub current_mode: ArmMode,
    pub regs: [u32; NUM_REGS],
    pub mem: MainMemory,
    tx: Sender<SystemMessage>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArmMode {
    ARM,
    THUMB,
}

pub const NUM_REGS: usize = 18;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    IP = 12,
    SP = 13,
    LR = 14,
    PC = 15,
    CPSR = 16,
    SPSR = 17,
}

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
    Reg::IP,
    Reg::SP,
    Reg::LR,
    Reg::PC,
    Reg::CPSR,
    Reg::SPSR,
];

// Need ability to pass mode as an i8 when calling from LLVM code
impl From<i8> for ArmMode {
    fn from(value: i8) -> Self {
        match value {
            0 => Self::ARM,
            1 => Self::THUMB,
            _ => panic!("invalid value for ArmMode: {}", value),
        }
    }
}

impl ArmMode {
    pub fn instr_size(&self) -> usize {
        match self {
            ArmMode::ARM => 4,
            ArmMode::THUMB => 2,
        }
    }
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
            ArmReg::ARM_REG_R12 => Reg::IP,
            ArmReg::ARM_REG_SP => Reg::SP,
            ArmReg::ARM_REG_LR => Reg::LR,
            ArmReg::ARM_REG_PC => Reg::PC,
            // Not 100% sure about this one
            ArmReg::ARM_REG_CPSR | ArmReg::ARM_REG_APSR | ArmReg::ARM_REG_SPSR => Reg::CPSR,
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

impl ArmMode {
    pub fn pc_byte_offset(&self) -> u32 {
        match self {
            ArmMode::ARM => 8,
            ArmMode::THUMB => 4,
        }
    }
}

impl ArmState {
    pub fn new(tx: Sender<SystemMessage>) -> Self {
        let mut regs = [0; NUM_REGS];
        regs[Reg::PC] = 8; // 2 instructions ahead
        Self {
            current_mode: ArmMode::ARM,
            regs,
            mem: MainMemory::default(),
            tx,
        }
    }

    pub fn curr_instr_addr(&self) -> usize {
        (self.regs[Reg::PC] - self.current_mode.pc_byte_offset()) as usize
    }

    pub fn jump_to(&mut self, addr: u32, mode: i8) {
        let new_mode = ArmMode::from(mode);
        if new_mode != self.current_mode {
            self.current_mode = new_mode;
            self.tx
                .send(SystemMessage::ChangeMode(new_mode))
                .expect("channel was closed");
        }
        self.regs[Reg::PC] = addr + self.current_mode.pc_byte_offset();
    }
}
