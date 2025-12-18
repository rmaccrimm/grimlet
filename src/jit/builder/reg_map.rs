use std::fmt::Pointer;

use inkwell::values::{IntValue, PointerValue};

use crate::arm::state::{NUM_REGS, Reg};

#[derive(Copy, Clone)]
pub struct RegMapItem<'a> {
    /// The current value of the register, to be written back to state object at end of function
    /// invocation if modified
    pub current_value: IntValue<'a>,

    /// Points the the associated element in the state register array
    pub state_ptr: PointerValue<'a>,

    /// whether value has changed or not
    pub dirty: bool,
}

pub struct RegMap<'a> {
    pub items: Vec<Option<RegMapItem<'a>>>,
}

#[allow(dead_code)]
impl<'a> RegMap<'a> {
    pub fn new() -> Self {
        Self {
            items: vec![None; NUM_REGS],
        }
    }

    pub fn init(&mut self, reg: Reg, value: IntValue<'a>, ptr: PointerValue<'a>) {
        self.items[reg as usize] = Some(RegMapItem {
            current_value: value,
            state_ptr: ptr,
            dirty: false,
        });
    }

    pub fn update(&mut self, reg: Reg, value: IntValue<'a>) {
        let item =
            self.items[reg as usize].unwrap_or_else(|| panic!("reg {:?} has not been loaded", reg));
        self.items[reg as usize] = Some(RegMapItem {
            current_value: value,
            state_ptr: item.state_ptr,
            dirty: true,
        });
    }

    pub fn get(&self, reg: Reg) -> IntValue<'a> {
        self.items[reg as usize]
            .unwrap_or_else(|| panic!("reg {:?} has not been loaded", reg))
            .current_value
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
