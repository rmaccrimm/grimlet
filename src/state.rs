#[repr(C)]
pub struct GuestState {
    pub regs: [u32; 16],
    pub mem: Box<[u8; 0xe010000]>,
}
