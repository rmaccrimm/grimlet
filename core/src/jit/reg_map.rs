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

#[derive(Clone)]
pub struct RegMap<'a> {
    pub items: Vec<Option<RegMapItem<'a>>>,
}

impl<'a> RegMap<'a> {
    pub fn new() -> Self {
        Self {
            items: vec![None; NUM_REGS as usize],
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
            self.items[reg as usize].unwrap_or_else(|| panic!("reg {reg:?} has not been loaded"));
        self.items[reg as usize] = Some(RegMapItem {
            current_value: value,
            state_ptr: item.state_ptr,
            dirty: true,
        });
    }

    pub fn get(&self, reg: Reg) -> IntValue<'a> {
        self.items[reg as usize]
            .unwrap_or_else(|| panic!("reg {reg:?} has not been loaded"))
            .current_value
    }
}
