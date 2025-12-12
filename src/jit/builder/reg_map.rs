use inkwell::values::IntValue;

use crate::arm::cpu::{NUM_REGS, Reg};

pub struct RegMap<'a> {
    pub llvm_values: Vec<Option<IntValue<'a>>>,
    pub dirty: Vec<bool>,
}

#[allow(dead_code)]
impl<'a> RegMap<'a> {
    pub fn new() -> Self {
        Self {
            llvm_values: vec![None; NUM_REGS],
            dirty: vec![false; NUM_REGS],
        }
    }

    pub fn update(&mut self, reg: Reg, value: IntValue<'a>) {
        self.llvm_values[reg as usize] = Some(value);
        self.dirty[reg as usize] = true;
    }

    pub fn get(&self, reg: Reg) -> IntValue<'a> {
        self.llvm_values[reg as usize].unwrap_or_else(|| panic!("reg {:?} was not loaded", reg))
    }

    pub fn r0(&self) -> IntValue<'a> { self.get(Reg::R0) }

    pub fn r1(&self) -> IntValue<'a> { self.get(Reg::R1) }

    pub fn r2(&self) -> IntValue<'a> { self.get(Reg::R2) }

    pub fn r3(&self) -> IntValue<'a> { self.get(Reg::R3) }

    pub fn r4(&self) -> IntValue<'a> { self.get(Reg::R4) }

    pub fn r5(&self) -> IntValue<'a> { self.get(Reg::R5) }

    pub fn r6(&self) -> IntValue<'a> { self.get(Reg::R6) }

    pub fn r7(&self) -> IntValue<'a> { self.get(Reg::R7) }

    pub fn r8(&self) -> IntValue<'a> { self.get(Reg::R8) }

    pub fn r9(&self) -> IntValue<'a> { self.get(Reg::R9) }

    pub fn r10(&self) -> IntValue<'a> { self.get(Reg::R10) }

    pub fn r11(&self) -> IntValue<'a> { self.get(Reg::R11) }

    pub fn ip(&self) -> IntValue<'a> { self.get(Reg::IP) }

    pub fn sp(&self) -> IntValue<'a> { self.get(Reg::SP) }

    pub fn lr(&self) -> IntValue<'a> { self.get(Reg::LR) }

    pub fn pc(&self) -> IntValue<'a> { self.get(Reg::PC) }

    pub fn cpsr(&self) -> IntValue<'a> { self.get(Reg::CPSR) }
}
