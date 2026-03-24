use std::array::TryFromSliceError;
use std::slice::Chunks;

use anyhow::{Result, bail};
use num::traits::{AsPrimitive, FromBytes, ToBytes};

#[derive(Clone, Debug)]
struct MemRegion {
    wait_states: u32,
    data: Vec<u8>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ReadVal(pub u32, pub u32);

pub struct MainMemory {
    bios: MemRegion,
    external_wram: MemRegion,
    internal_wram: MemRegion,
    io_registers: MemRegion,
    palette_ram: MemRegion,
    vram: MemRegion,
    obj_attrs: MemRegion,
    cartridge_rom: MemRegion,
}

// TODO - will add these as needed
pub enum IoReg {
    DISPCNT = 0x4000000,
    DISPSTAT = 0x4000004,
}

impl MemRegion {
    fn new(size: usize, wait_states: u32) -> Self {
        MemRegion {
            wait_states,
            data: vec![0; size],
        }
    }
}

impl MainMemory {
    pub fn iter_word(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(4) {
            panic!("Mis-alligned word address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.0.chunks(4))
    }

    pub fn iter_halfword(&self, start_addr: u32) -> Result<Chunks<'_, u8>> {
        if !start_addr.is_multiple_of(2) {
            panic!("Mis-alligned halfword address: {:x}", start_addr);
        }
        Ok(self.mem_map_lookup(start_addr)?.0.chunks(2))
    }

    /// Sign or zero-extends the result to 32 bits depending type parameter
    /// Returns both the read value and the number of wait-states
    pub fn read<T>(&self, addr: u32) -> ReadVal
    where
        T: AsPrimitive<u32>
            + for<'a> FromBytes<Bytes: TryFrom<&'a [u8], Error = TryFromSliceError>>,
    {
        let (mem_slice, wait_states) = self.mem_map_lookup(addr).expect("out of bounds read");
        let bytes: T::Bytes = mem_slice
            .get(0..size_of::<T>())
            .expect("reached end of bytes while reading")
            .try_into()
            .expect("conversion from bytes failed");
        let as_int = T::from_le_bytes(&bytes);
        ReadVal(as_int.as_(), wait_states)
    }

    pub fn write<T>(&mut self, addr: u32, value: T) -> u32
    where
        T: ToBytes<Bytes: IntoIterator<Item = u8>>,
    {
        let (mem, wait_states) = self.mem_map_lookup_mut(addr).expect("out of bounds write");
        let mut mem_iter = mem.iter_mut();
        for byte in value.to_le_bytes() {
            let mem_val = mem_iter.next().expect("reached end of bytes while writing");
            *mem_val = byte;
        }
        wait_states
    }

    // TODO - should these be public?
    pub fn mem_map_lookup(&self, addr: u32) -> Result<(&[u8], u32)> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        let region = match base {
            0x00000000 => &self.bios,
            0x02000000 => &self.external_wram,
            0x03000000 => &self.internal_wram,
            0x04000000 => &self.io_registers,
            0x05000000 => &self.palette_ram,
            0x06000000 => &self.vram,
            0x07000000 => &self.obj_attrs,
            0x08000000 => &self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        };
        Ok((&region.data[index..], region.wait_states))
    }

    // Would be nice if these could be combined somehow
    pub fn mem_map_lookup_mut(&mut self, addr: u32) -> Result<(&mut [u8], u32)> {
        let base = addr & 0x0f000000;
        let index = (addr - base) as usize;
        let region = match base {
            0x00000000 => &mut self.bios,
            0x02000000 => &mut self.external_wram,
            0x03000000 => &mut self.internal_wram,
            0x04000000 => &mut self.io_registers,
            0x05000000 => &mut self.palette_ram,
            0x06000000 => &mut self.vram,
            0x07000000 => &mut self.obj_attrs,
            0x08000000 => &mut self.cartridge_rom,
            _ => bail!("unused area of memory (addr: {:#08x})", addr),
        };
        Ok((&mut region.data[index..], region.wait_states))
    }

    // Returns (IO register, wait-states).
    // Not all IO registers are actually 32-bits wide. Leave that up to the caller
    pub fn read_io(&self, reg: IoReg) -> ReadVal { self.read::<u32>(reg as u32) }
}

impl Default for MainMemory {
    fn default() -> Self {
        Self {
            bios: MemRegion::new(16 << 10, 0),
            external_wram: MemRegion::new(256 << 10, 0),
            internal_wram: MemRegion::new(32 << 10, 0),
            io_registers: MemRegion::new(0x400, 0),
            palette_ram: MemRegion::new(1 << 10, 0),
            vram: MemRegion::new(96 << 10, 0),
            obj_attrs: MemRegion::new(1 << 10, 0),
            // TODO -  wait states configurable? Possibly seperate struct, maybe a Readable trait?
            cartridge_rom: MemRegion::new(32 << 20, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! read_tests {
        ($bytes:expr, $($name:ident: $T:ty, $data:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (read_addr, expected) = $data;

                    let mut mem = MainMemory::default();
                    mem.bios = MemRegion{data: $bytes, wait_states: 0};
                    assert_eq!(mem.read::<$T>(read_addr).0, expected);
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
            bios: MemRegion {
                data: vec![0x34, 0xff, 0xbe, 0x70],
                wait_states: 0,
            },
            ..Default::default()
        };
        mem.read::<u32>(1);
    }

    #[test]
    fn test_write() {
        let mut mem = MainMemory {
            bios: MemRegion::new(4, 0),
            ..Default::default()
        };
        mem.write(0, 0x12345678u32);
        assert_eq!(mem.bios.data, vec![0x78, 0x56, 0x34, 0x12]);
        mem.write(0, 0u8);
        assert_eq!(mem.bios.data, vec![0x00, 0x56, 0x34, 0x12]);
        mem.write(1, 0u16);
        assert_eq!(mem.bios.data, vec![0x00, 0x00, 0x00, 0x12]);
        mem.write(0, 0x3102u16);
        assert_eq!(mem.bios.data, vec![0x02, 0x31, 0x00, 0x12]);
        mem.write(0, -12i8);
        assert_eq!(mem.bios.data, vec![0xf4, 0x31, 0x00, 0x12]);
        mem.write(2, 0xabu8);
        assert_eq!(mem.bios.data, vec![0xf4, 0x31, 0xab, 0x12]);
        mem.write(0, -10500i16);
        assert_eq!(mem.bios.data, vec![0xfc, 0xd6, 0xab, 0x12]);
    }
}
