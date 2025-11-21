use anyhow::{Result, anyhow};
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::StructType;
use std::fs;
use std::fs::File;
use std::io::Read;

#[repr(C)]
pub struct MainMemory {
    pub bios: Vec<u8>,
}

impl MainMemory {
    pub fn new() -> Self {
        // 16 kB
        let bios = vec![0; 0x4000];
        Self { bios }
    }
}

pub enum ArmMode {
    ARM,
    THUMB,
}

#[derive(Copy, Clone, Debug)]
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

pub const NUM_REGS: usize = 17;

/// Emulated CPU state and interpreter
#[repr(C)]
pub struct ArmState {
    pub mode: ArmMode,
    pub regs: [u32; 17],
    pub mem: Box<MainMemory>,
}

impl ArmState {
    pub fn new() -> Self {
        Self {
            mode: ArmMode::ARM,
            regs: [0; NUM_REGS],
            mem: Box::new(MainMemory::new()),
        }
    }

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
        let mut st = ArmState::new();

        if !fs::exists(bios_path)? {
            return Err(anyhow!("BIOS file not found"));
        }
        let mut f = File::open(bios_path)?;
        f.read_to_end(&mut st.mem.bios)?;

        Ok(st)
    }

    pub fn pc(&self) -> u32 {
        self.regs[Reg::PC as usize]
    }

    pub fn jump_to(&mut self, addr: u32) {
        println!("JUMPING TO: {}", addr);
        self.regs[Reg::PC as usize] = addr;
    }
}
