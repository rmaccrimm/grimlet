use anyhow::{Result, anyhow};
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

/// Emulated CPU state and interpreter
#[repr(C)]
pub struct ArmState {
    pub regs: [u32; 17],
    // pub mem: Box<[u8; 0x4000]>,
    pub mem: Vec<u8>,
}

impl ArmState {
    pub fn new() -> Self {
        let regs = [0; 17];
        let mem = vec![0; 0x4000];
        Self { regs, mem }
    }

    pub fn with_bios(bios_path: &str) -> Result<Self> {
        let regs = [0; 17];
        let mut mem = Vec::new();

        if !fs::exists(bios_path)? {
            return Err(anyhow!("BIOS file not found"));
        }
        let mut f = File::open(bios_path)?;
        f.read_to_end(&mut mem)?;

        Ok(Self { regs, mem })
    }

    pub fn pc(&self) -> u32 {
        self.regs[Reg::PC as usize]
    }

    pub fn jump_to(&mut self, addr: u32) {
        println!("JUMPING TO: {}", addr);
        self.regs[Reg::PC as usize] = addr;
    }
}
