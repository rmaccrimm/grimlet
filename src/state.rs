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

#[repr(C)]
pub struct GuestState {
    pub regs: [u32; 17],
    pub mem: Box<[u8; 0x4000]>,
}

impl GuestState {
    pub fn new() -> Self {
        let mem = Box::new([0; 0x4000]);
        let regs = [0; 17];
        Self { regs, mem }
    }

    pub fn pc(&self) -> &u32 {
        &self.regs[15]
    }
}
