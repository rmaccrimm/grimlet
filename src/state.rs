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

#[repr(C)]
pub struct GuestState {
    pub regs: [u32; 16],
    pub mem: MainMemory,
}

impl GuestState {
    pub fn new() -> Self {
        let mem = MainMemory::new();
        let regs = [0; 16];
        Self { regs, mem }
    }

    pub fn pc(&self) -> &u32 {
        &self.regs[15]
    }
}
