use std::array::TryFromSliceError;
use std::slice::Chunks;

use num::traits::{AsPrimitive, FromBytes, ToBytes};

#[repr(C)]
pub struct MainMemory {
    pub bios: Vec<u8>,
}

pub trait MemReadable =
    AsPrimitive<u32> + for<'a> FromBytes<Bytes: TryFrom<&'a [u8], Error = TryFromSliceError>>;

pub trait MemWriteable = ToBytes<Bytes: IntoIterator<Item = u8>>;

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

    /// Sign or zero-extends the result to 32 bits depending type parameter
    pub fn read<T>(&self, addr: u32) -> u32
    where
        T: MemReadable,
    {
        let mem_slice = self
            .mem_map_lookup(addr)
            .get(0..size_of::<T>())
            .expect("reached end of bytes while reading");
        let bytes: T::Bytes = mem_slice.try_into().expect("conversion from bytes failed");
        let as_int = T::from_le_bytes(&bytes);
        as_int.as_()
    }

    pub fn write<T>(&mut self, addr: u32, value: T)
    where
        T: MemWriteable,
    {
        let mut mem_iter = self.mem_map_lookup_mut(addr).iter_mut();
        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("reached end of bytes while writing");
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

#[cfg(test)]
mod tests {
    use crate::arm::state::memory::MainMemory;

    #[test]
    fn test_read_unsigned() {
        let mem = MainMemory {
            bios: vec![0x34, 0xff, 0xbe, 0x70, 0xf1],
        };
        // // TODO enforce alignment?
        assert_eq!(mem.read::<u8>(0), 0x00000034);
        assert_eq!(mem.read::<u8>(1), 0x000000ff);
        assert_eq!(mem.read::<u8>(2), 0x000000be);
        assert_eq!(mem.read::<u16>(0), 0x0000ff34);
        assert_eq!(mem.read::<u16>(1), 0x0000beff);
        assert_eq!(mem.read::<u16>(2), 0x000070be);
        assert_eq!(mem.read::<u32>(0), 0x70beff34);
        assert_eq!(mem.read::<u32>(1), 0xf170beff);
    }

    #[test]
    fn test_read_signed() {
        let mem = MainMemory {
            bios: vec![0x84, 0xff, 0x3e, 0x70, 0x80],
        };
        assert_eq!(mem.read::<i8>(0), 0xffffff84);
        assert_eq!(mem.read::<i8>(2), 0x0000003e);
        assert_eq!(mem.read::<i16>(0), 0xffffff84);
        assert_eq!(mem.read::<i16>(1), 0x00003eff);
        assert_eq!(mem.read::<i32>(0), 0x703eff84);
        assert_eq!(mem.read::<i32>(1), 0x80703eff);
    }

    #[test]
    #[should_panic = "reached end of bytes while reading"]
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
        mem.write(0, 0u8);
        assert_eq!(mem.bios, vec![0x00, 0x56, 0x34, 0x12]);
        mem.write(1, 0u16);
        assert_eq!(mem.bios, vec![0x00, 0x00, 0x00, 0x12]);
        mem.write(0, 0x3102u16);
        assert_eq!(mem.bios, vec![0x02, 0x31, 0x00, 0x12]);
        mem.write(0, -12i8);
        assert_eq!(mem.bios, vec![0xf4, 0x31, 0x00, 0x12]);
        mem.write(2, 0xabu8);
        assert_eq!(mem.bios, vec![0xf4, 0x31, 0xab, 0x12]);
        mem.write(0, -10500i16);
        assert_eq!(mem.bios, vec![0xfc, 0xd6, 0xab, 0x12]);
    }
}
