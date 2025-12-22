use std::array::TryFromSliceError;
use std::slice::Chunks;

use anyhow::{Result, bail};
use num::traits::{AsPrimitive, FromBytes, ToBytes};

#[repr(C)]
pub struct MainMemory {
    bios: Vec<u8>,
    cart_rom: Vec<u8>,
    io_registers: Vec<u8>,
}

pub trait MemReadable =
    AsPrimitive<u32> + for<'a> FromBytes<Bytes: TryFrom<&'a [u8], Error = TryFromSliceError>>;

pub trait MemWriteable = ToBytes<Bytes: IntoIterator<Item = u8>>;

impl MainMemory {
    pub fn iter_word(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(4) {
            panic!("Mis-alligned word address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.chunks(4))
    }

    pub fn iter_halfword(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(2) {
            panic!("Mis-alligned halfword address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.chunks(2))
    }

    /// Sign or zero-extends the result to 32 bits depending type parameter
    pub fn read<T>(&self, addr: u32) -> u32
    where
        T: MemReadable,
    {
        let mem_slice = self
            .mem_map_lookup(addr)
            .expect("out of bounds read")
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
        let mut mem_iter = self
            .mem_map_lookup_mut(addr)
            .expect("out of bounds write")
            .iter_mut();
        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("reached end of bytes while writing");
            *mem_val = byte;
        }
    }

    pub fn get_ptr(&self, addr: u32) -> *const u8 {
        self.mem_map_lookup(addr)
            .expect("get_ptr address out of bounds")
            .as_ptr()
    }

    /// Planning to use this to be slightly more convenient for multiple stores. TODO what if
    /// they were writing to memory mapped IO? Could that happen?
    pub fn get_mut_ptr(&mut self, addr: u32) -> *mut u8 {
        self.mem_map_lookup_mut(addr)
            .expect("get_mut_ptr address out of bounds")
            .as_mut_ptr()
    }

    pub fn mem_map_lookup(&self, addr: u32) -> Result<&[u8]> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        Ok(match base {
            0x00000000 => &self.bios[index..],
            0x04000000 => &self.io_registers[index..],
            0x08000000 => &self.cart_rom[index..],
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        })
    }

    // Would be nice if these could be combined somehow
    pub fn mem_map_lookup_mut(&mut self, addr: u32) -> Result<&mut [u8]> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        Ok(match base {
            0x00000000 => &mut self.bios[index..],
            0x04000000 => &mut self.io_registers[index..],
            0x08000000 => &mut self.cart_rom[index..],
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        })
    }
}

impl Default for MainMemory {
    fn default() -> Self {
        Self {
            bios: vec![0; 16 << 10],
            cart_rom: vec![0; 1 << 10], // TODO actual size
            io_registers: vec![0; 0x3ff],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arm::state::memory::MainMemory;

    macro_rules! read_tests {
        ($bytes:expr, $($name:ident: $T:ty, $data:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (read_addr, expected) = $data;

                    let mem = MainMemory {
                        bios: $bytes,
                        io_registers: vec![],
                        cart_rom: vec![],
                    };
                    assert_eq!(mem.read::<$T>(read_addr), expected);
                }
            )*
        };
    }

    read_tests! {
        vec![0x34, 0xff, 0xbe, 0x70, 0xf1],
        test_read_unsigned_0: u8, (0, 0x00000034),
        test_read_unsigned_1: u8, (1, 0x000000ff),
        test_read_unsigned_2: u8, (2, 0x000000be),
        test_read_unsigned_3: u16, (0, 0x0000ff34),
        test_read_unsigned_4: u16, (1, 0x0000beff),
        test_read_unsigned_5: u16, (2, 0x000070be),
        test_read_unsigned_6: u32, (0, 0x70beff34),
        test_read_unsigned_7: u32, (1, 0xf170beff),
    }

    read_tests! {
        vec![0x84, 0xff, 0x3e, 0x70, 0x80],
        test_read_signed_0: i8, (0, 0xffffff84),
        test_read_signed_1: i8, (2, 0x0000003e),
        test_read_signed_2: i16, (0, 0xffffff84),
        test_read_signed_3: i16, (1, 0x00003eff),
        test_read_signed_4: i32, (0, 0x703eff84),
        test_read_signed_5: i32, (1, 0x80703eff),
    }

    #[test]
    #[should_panic = "reached end of bytes while reading"]
    fn test_read_past_end_of_bytes_panics() {
        let mem = MainMemory {
            bios: vec![0x34, 0xff, 0xbe, 0x70],
            io_registers: vec![],
            cart_rom: vec![],
        };
        mem.read::<u32>(1);
    }

    #[test]
    fn test_write() {
        let mut mem = MainMemory {
            bios: vec![0; 4],
            io_registers: vec![],
            cart_rom: vec![],
        };
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
