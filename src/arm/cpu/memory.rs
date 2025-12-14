use std::slice::Chunks;

#[repr(C)]
pub struct MainMemory {
    pub bios: Vec<u8>,
}

pub trait TryFromLeSlice {
    fn try_from_le_slice(bytes: &[u8]) -> Result<Self, String>
    where
        Self: Sized;
}

pub trait ToLeBytes {
    type Bytes: IntoIterator<Item = u8>;

    fn to_le_bytes(self) -> Self::Bytes;
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

    pub fn read<T>(&self, addr: u32) -> T
    where
        T: TryFromLeSlice,
    {
        let mem_slice = self.mem_map_lookup(addr);
        TryFromLeSlice::try_from_le_slice(mem_slice).expect("failed to read bytes")
    }

    pub fn read_u32(&self, addr: u32) -> u32 {
        let mem_slice = self.mem_map_lookup(addr);
        u32::try_from_le_slice(mem_slice).expect("read failed")
    }

    pub fn write(&mut self, addr: u32, value: impl ToLeBytes) {
        let mut mem_iter = self.mem_map_lookup_mut(addr).iter_mut();

        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("hit end of mem segment");
            *mem_val = byte;
        }
    }

    pub fn get_ptr(&self, addr: u32) -> *const u8 { self.mem_map_lookup(addr).as_ptr() }

    /// Planning to use this to be slightly more convenient for multiple stores. TODO what if
    /// they were writing to memory mapped IO? Could that happen?
    pub fn get_mut_ptr(&mut self, addr: u32) -> *mut u8 {
        self.mem_map_lookup_mut(addr).as_mut_ptr()
    }

    fn mem_map_lookup(&self, addr: u32) -> &[u8] {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        match base {
            0x00000000 => &self.bios[index..],
            _ => panic!("unused area of memory (addr: {})", addr),
        }
    }

    // Would be nice if these could be combined somehow
    fn mem_map_lookup_mut(&mut self, addr: u32) -> &mut [u8] {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        match base {
            0x00000000 => &mut self.bios[index..],
            _ => panic!("unused area of memory (addr: {})", addr),
        }
    }
}

macro_rules! impl_try_from_le_slice {
    ($T:ty) => {
        impl TryFromLeSlice for $T {
            fn try_from_le_slice(bytes: &[u8]) -> Result<Self, String> {
                let arr = bytes
                    .get(..{ size_of::<$T>() })
                    .ok_or(format!(
                        "reached end of bytes while reading {}",
                        stringify!($T)
                    ))?
                    .try_into()
                    .map_err(|e| format!("{}", e))?;
                Ok(Self::from_le_bytes(arr))
            }
        }
    };
}

macro_rules! impl_to_le_bytes {
    ($T:ty) => {
        impl ToLeBytes for $T {
            type Bytes = [u8; { size_of::<$T>() }];

            fn to_le_bytes(self) -> Self::Bytes { <$T>::to_le_bytes(self) }
        }
    };
}

impl_try_from_le_slice!(u8);
impl_try_from_le_slice!(u16);
impl_try_from_le_slice!(u32);
impl_to_le_bytes!(u8);
impl_to_le_bytes!(u16);
impl_to_le_bytes!(u32);

#[cfg(test)]
mod tests {
    use crate::arm::cpu::memory::MainMemory;

    #[test]
    fn test_read() {
        let mem = MainMemory {
            bios: vec![0x34, 0xff, 0xbe, 0x70],
        };
        assert_eq!(mem.read::<u8>(0), 0x34);
        assert_eq!(mem.read::<u16>(0), 0xff34);
        assert_eq!(mem.read::<u32>(0), 0x70beff34);
        // TODO enforce alignment?
        assert_eq!(mem.read::<u8>(1), 0xff);
        assert_eq!(mem.read::<u16>(1), 0xbeff);
        assert_eq!(mem.read::<u8>(2), 0xbe);
        assert_eq!(mem.read::<u16>(2), 0x70be);
    }

    #[test]
    #[should_panic = "reached end of bytes while reading u32"]
    fn test_read_past_end_of_bytes_panics() {
        let mem = MainMemory {
            bios: vec![0x34, 0xff, 0xbe, 0x70],
        };
        mem.read::<u32>(1);
    }

    #[test]
    fn test_write() {
        let mut mem = MainMemory { bios: vec![0; 4] };
        mem.write(0, 0x12345678u32);
        assert_eq!(mem.bios, vec![0x78, 0x56, 0x34, 0x12]);
    }
}
